# med_assist.py

import os
import asyncio
from dotenv import load_dotenv

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import CancellationToken

# We'll import the classification prompt & specialist configs
from prompts_config import INTENT_CLASSIFICATION_TEMPLATE, SPECIALIST_CONFIGS

# We'll import the VoiceUI class
from voice_ui import VoiceUI

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VER  = "2024-06-01"

# We'll use GPT-4o for both classification + specialists
GPT4O_DEPLOYMENT = "gpt-4o"

class ClassificationAndSpecialistTeam:
    """
    1) Classify user text with an agent using INTENT_CLASSIFICATION_TEMPLATE
    2) pick a specialist config from SPECIALIST_CONFIGS
    3) return final text
    """
    def __init__(self):
        self.model_client = AzureOpenAIChatCompletionClient(
            model=GPT4O_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VER,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )

        # The classification agent
        self.classifier_agent = AssistantAgent(
            name="ClassifierAgent",
            model_client=self.model_client,
            system_message="You are an intent classification agent."
        )

    async def handle_user_message(self, user_text: str) -> str:
        # Step A: Classification using INTENT_CLASSIFICATION_TEMPLATE
        prompt_text = INTENT_CLASSIFICATION_TEMPLATE.format(user_message=user_text)
        response = await self.classifier_agent.on_messages(
            [TextMessage(content=prompt_text, source="user")],
            cancellation_token=CancellationToken()
        )
        classification = response.chat_message.content.strip().lower()
        print(f"[ClassifierAgent => '{classification}']")

        # Step B: Pick specialist config
        if classification not in SPECIALIST_CONFIGS:
            classification = "general_medical"
        config = SPECIALIST_CONFIGS[classification]

        # Step C: Create specialist agent on the fly
        spec_agent = AssistantAgent(
            name=config["name"],
            model_client=self.model_client,
            system_message=config["system_message"]
        )

        # (Optionally short-circuit if classification == 'emergency')
        # We'll let the specialist agent handle it so it can say "call 911" etc

        # Step D: The user text is still the same
        spec_response = await spec_agent.on_messages(
            [TextMessage(content=user_text, source="user")],
            cancellation_token=CancellationToken()
        )
        return spec_response.chat_message.content

async def main():
    # 1) Build the classification+specialist team
    team = ClassificationAndSpecialistTeam()

    # 2) Create voice UI
    voice_ui = VoiceUI()
    await voice_ui.initialize()

    print("Voice-based medical triage assistant is active. Speak into mic...")

    try:
        while True:
            # a) Listen for user speech
            user_text = await voice_ui.listen()
            if not user_text:
                continue

            print(f"[User said: {user_text}]")

            # b) get the final answer
            final_answer = await team.handle_user_message(user_text)
            print(f"[Assistant => {final_answer}]")

            # c) speak it
            await voice_ui.speak(final_answer)

    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        await voice_ui.close()

if __name__ == "__main__":
    asyncio.run(main())
