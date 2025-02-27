import os
import asyncio
from dotenv import load_dotenv

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import CancellationToken

# Import classification prompt & specialist configs
from prompts_config import INTENT_CLASSIFICATION_TEMPLATE, SPECIALIST_CONFIGS

# Import the updated VoiceUI class
from voice_ui import VoiceUI

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VER  = "2024-06-01"

# We'll use GPT-4o for both classification and specialist agents
GPT4O_DEPLOYMENT = "gpt-4o"

class ClassificationAndSpecialistTeam:
    """
    1) Classify user text using INTENT_CLASSIFICATION_TEMPLATE.
    2) Select a specialist config from SPECIALIST_CONFIGS.
    3) Return the final text response.
    """
    def __init__(self):
        self.model_client = AzureOpenAIChatCompletionClient(
            model=GPT4O_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VER,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        self.classifier_agent = AssistantAgent(
            name="ClassifierAgent",
            model_client=self.model_client,
            system_message="You are an intent classification agent."
        )

    async def handle_user_message(self, user_text: str) -> str:
        # Step A: Classify using the provided template.
        prompt_text = INTENT_CLASSIFICATION_TEMPLATE.format(user_message=user_text)
        response = await self.classifier_agent.on_messages(
            [TextMessage(content=prompt_text, source="user")],
            cancellation_token=CancellationToken()
        )
        classification = response.chat_message.content.strip().lower()
        print(f"[ClassifierAgent => '{classification}']")

        # Step B: Select the specialist config.
        if classification not in SPECIALIST_CONFIGS:
            classification = "general_medical"
        config = SPECIALIST_CONFIGS[classification]

        # Step C: Create the specialist agent on the fly.
        spec_agent = AssistantAgent(
            name=config["name"],
            model_client=self.model_client,
            system_message=config["system_message"]
        )

        # Step D: Let the specialist agent process the original user text.
        spec_response = await spec_agent.on_messages(
            [TextMessage(content=user_text, source="user")],
            cancellation_token=CancellationToken()
        )
        return spec_response.chat_message.content

async def main():
    # Build the classification and specialist team.
    team = ClassificationAndSpecialistTeam()

    # Initialize the updated multi-turn VoiceUI.
    voice_ui = VoiceUI()
    await voice_ui.initialize()

    print("Voice-based medical triage assistant is active. Speak into mic...")

    try:
        while True:
            # Listen for user speech.
            user_text = await voice_ui.listen()
            if not user_text:
                continue

            # Optional: allow exit via voice command.
            if user_text.lower().strip() in ["exit", "quit"]:
                print("Exit command received. Exiting.")
                break

            print(f"[User said: {user_text}]")

            # Get final answer from the classification and specialist team.
            final_answer = await team.handle_user_message(user_text)
            print(f"[Assistant => {final_answer}]")

            # Voice out the answer.
            await voice_ui.speak(final_answer)

    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        await voice_ui.close()

if __name__ == "__main__":
    asyncio.run(main())
