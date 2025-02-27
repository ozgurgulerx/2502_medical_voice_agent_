# intent_classifier_agent.py

from autogen import AssistantAgent

class IntentClassifierAgent(AssistantAgent):
    """
    An AutoGen-based agent that uses an LLM to classify
    user text into an intent category.
    """
    def __init__(self, name="IntentClassifierAgent", **kwargs):
        # The system prompt sets the context for classification
        system_prompt = (
            "You are an intent classification agent for a medical call.\n"
            "You have three possible intent categories:\n"
            "1) NEW_SYMPTOM: If the user is describing new symptoms.\n"
            "2) EXISTING_CONDITION: If the user references an ongoing or existing health issue.\n"
            "3) PATIENT_ID: If the user references or supplies a patient ID number (e.g., 'my ID is 1234').\n"
            "4) UNKNOWN: If none of the above apply.\n"
            "Respond ONLY with one category name: NEW_SYMPTOM, EXISTING_CONDITION, PATIENT_ID, or UNKNOWN.\n"
        )
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            **kwargs
        )

    def classify_intent(self, user_text: str) -> str:
        """
        Calls the LLM with a direct prompt. 
        Expects a single label from the agent's reply.
        """
        # We'll just prompt it with the user's text + instructions.
        # A real system might do few-shot examples or function calling for more robust parsing.
        prompt = (
            f"User says: \"{user_text}\"\n"
            "Decide the correct category: NEW_SYMPTOM, EXISTING_CONDITION, PATIENT_ID, or UNKNOWN.\n"
            "Answer with ONLY the category name."
        )
        response = self.step(prompt)
        # The LLM might add extra text, so let's be safe and parse the first word
        # Or do any other minimal cleanup logic
        category = response["content"].strip().upper()
        # Just in case the LLM gave a longer answer or something
        if "EXISTING" in category:
            return "EXISTING_CONDITION"
        elif "PATIENT" in category or "ID" in category:
            return "PATIENT_ID"
        elif "NEW" in category:
            return "NEW_SYMPTOM"
        else:
            return "UNKNOWN"
