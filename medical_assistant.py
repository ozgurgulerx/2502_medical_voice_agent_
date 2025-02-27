import os
import autogen
from autogen import UserProxyAgent, AssistantAgent
from openai import AzureOpenAI
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

# Import your existing Voice UI
from voice_ui import VoiceUI
# Import configurations
from config import SPECIALIST_CONFIGS, INTENT_CLASSIFICATION_TEMPLATE

# Load environment variables
load_dotenv()

# Configure Azure OpenAI client
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError("Missing Azure OpenAI credentials in .env file")

azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-06-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

class GatekeeperAgent(AssistantAgent):
    """
    An agent that uses o1-mini to classify user intent and route to appropriate specialists.
    """
    def __init__(self, name="Gatekeeper", 
                 system_message="You are a medical triage assistant that classifies user intents.", 
                 **kwargs):
        super().__init__(
            name=name,
            system_message=system_message,
            # Note: We'll override the LLM call with Azure OpenAI
            llm_config={"model": "o1-mini"},
            **kwargs
        )
        self.intent_categories = [
            "new_symptoms", 
            "existing_condition", 
            "general_medical", 
            "emergency"
        ]
        
    async def _classify_intent(self, message: str) -> str:
        """
        Use Azure OpenAI's o1-mini to classify the user's intent.
        """
        prompt = f"""
        Analyze the following medical query and classify it into exactly ONE of these categories:
        1. new_symptoms: User is seeking care for new symptoms unrelated to existing conditions
        2. existing_condition: User is seeking care for an existing condition
        3. general_medical: User is seeking care for other general medical reasons
        4. emergency: User is describing a medical emergency
        
        Provide careful step-by-step reasoning before making your final classification.
        
        USER QUERY: {message}
        
        Think step by step:
        """
        
        # Use Azure OpenAI client instead of direct OpenAI
        response = azure_client.chat.completions.create(
            model="o1-mini",  # Use the appropriate Azure deployment name
            messages=[
                {"role": "system", "content": "You are a medical intent classifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # Extract reasoning and classification from response
        full_response = response.choices[0].message.content
        
        # Simple extraction - in practice would need more robust parsing
        if "emergency" in full_response.lower():
            return "emergency"
        elif "existing_condition" in full_response.lower():
            return "existing_condition"
        elif "new_symptoms" in full_response.lower():
            return "new_symptoms"
        else:
            return "general_medical"
    
    async def generate_reply(self, message: str, sender: Any, **kwargs) -> Optional[str]:
        """
        Override to classify intent and route to specialized agents.
        """
        intent = await self._classify_intent(message)
        
        # If emergency, provide immediate response
        if intent == "emergency":
            emergency_response = "I've detected this may be a medical emergency. " \
                               "Please call emergency services (911) immediately. " \
                               "Do not wait for further assistance through this system."
            return emergency_response
            
        # For non-emergencies, route to appropriate specialist agent
        # The actual routing would happen in the MedicalAssistant class
        return f"INTENT:{intent}\n{message}"


class SpecialistAgent(AssistantAgent):
    """
    Base class for specialist medical agents that handle specific intents.
    """
    def __init__(self, name: str, specialty: str, **kwargs):
        system_message = f"""You are a medical assistant specializing in {specialty}.
        Provide helpful, accurate medical information within your specialty area.
        Always clarify you are an AI and cannot provide official medical diagnosis.
        Keep responses conversational and appropriate for voice interaction."""
        
        # Create custom LLM config that uses the Azure client
        azure_llm_config = {
            "config_list": [{"model": "gpt-4o"}],  # Use your Azure deployment name
            "azure_client": azure_client
        }
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=azure_llm_config,
            **kwargs
        )
    
    async def generate_reply(self, message: str, sender: Any, **kwargs) -> Optional[str]:
        """Override generate_reply to use Azure OpenAI directly instead of through AutoGen"""
        response = azure_client.chat.completions.create(
            model="gpt-4o",  # Use your Azure deployment name
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content


class MedicalAssistant:
    """
    Orchestrates the entire medical assistant system, connecting voice UI with AutoGen agents.
    """
    def __init__(self):
        # Initialize voice UI
        self.voice_ui = VoiceUI()
        
        # Create user proxy agent
        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",  # We'll provide messages from voice UI
            max_consecutive_auto_reply=0
        )
        
        # Create gatekeeper agent
        self.gatekeeper = GatekeeperAgent()
        
        # Create specialist agents
        self.specialists = {
            "new_symptoms": SpecialistAgent(
                name="NewSymptomsSpecialist",
                specialty="evaluating new medical symptoms"
            ),
            "existing_condition": SpecialistAgent(
                name="ExistingConditionSpecialist",
                specialty="managing existing medical conditions"
            ),
            "general_medical": SpecialistAgent(
                name="GeneralMedicalSpecialist",
                specialty="general medical information and healthcare guidance"
            ),
            "emergency": SpecialistAgent(
                name="EmergencySpecialist",
                specialty="medical emergencies and urgent care triage"
            )
        }
        
    async def process_user_input(self, user_message: str) -> str:
        """
        Process user input from voice UI through the agent system and return response.
        """
        # Send message to gatekeeper for classification
        gatekeeper_response = await self.gatekeeper.generate_reply(
            message=user_message,
            sender=self.user_proxy
        )
        
        # Handle emergency responses immediately
        if gatekeeper_response.startswith("I've detected this may be a medical emergency"):
            return gatekeeper_response
            
        # Parse intent from gatekeeper response
        intent_line, modified_message = gatekeeper_response.split("\n", 1)
        intent = intent_line.replace("INTENT:", "").strip()
        
        # Get appropriate specialist
        specialist = self.specialists.get(intent, self.specialists["general_medical"])
        
        # Get response from specialist
        specialist_response = await specialist.generate_reply(
            message=modified_message,
            sender=self.user_proxy
        )
        
        return specialist_response
    
    async def start_conversation(self):
        """
        Initiate and manage a conversation through the voice UI.
        """
        # Initialize voice UI
        await self.voice_ui.initialize()
        
        try:
            # Voice UI greets user
            greeting = "Hello, I'm your medical assistant. How can I help you today?"
            await self.voice_ui.speak(greeting)
            
            while True:
                # Get user input through voice UI
                user_message = await self.voice_ui.listen()
                
                # Check for end of conversation
                if "goodbye" in user_message.lower() or "thank you" in user_message.lower():
                    closing = "Thank you for using our medical assistant. Take care and goodbye."
                    await self.voice_ui.speak(closing)
                    break
                    
                # Process through agents
                response = await self.process_user_input(user_message)
                
                # Convert response to speech
                await self.voice_ui.speak(response)
        finally:
            # Ensure proper cleanup
            await self.voice_ui.close()


async def main():
    """
    Main entry point for the application.
    """
    assistant = MedicalAssistant()
    await assistant.start_conversation()


if __name__ == "__main__":
    asyncio.run(main())