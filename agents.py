# agents.py
import os
# Suppose in autogen_primer.py we have example utility code or examples 
# We'll just mention them here as if we read them or imported them
from autogen import AssistantAgent

# Basic sample data store for demonstration
PATIENT_RECORDS = {
    "1234": {
        "name": "Alice Johnson",
        "conditions": ["chronic migraine"],
        "medications": ["Sumatriptan", "Propranolol"]
    },
    "2345": {
        "name": "Bob Anderson",
        "conditions": ["hypertension", "type-2 diabetes"],
        "medications": ["Lisinopril", "Metformin"]
    }
}

class SymptomsAgent(AssistantAgent):
    """
    AutoGen-based agent that handles new symptoms.
    We rely on the system_prompt to shape the agent's 'persona'.
    """
    def __init__(self, name="SymptomsAgent", **kwargs):
        system_prompt = (
            "You are a highly specialized triage agent focusing on new symptoms.\n"
            "Ask clarifying questions about onset, duration, severity.\n"
            "If the user references an existing condition or patient ID, politely mention that they should provide it.\n"
        )
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            **kwargs
        )


class HistoryAgent(AssistantAgent):
    """
    AutoGen-based agent that looks up patient history when given a valid patient ID.
    """
    def __init__(self, name="HistoryAgent", **kwargs):
        system_prompt = (
            "You are a medical records agent with access to patient records.\n"
            "When the user provides a patient ID, retrieve conditions and medications from PATIENT_RECORDS.\n"
            "If no record is found, say so politely.\n"
        )
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            **kwargs
        )

    def retrieve_record(self, patient_id: str) -> str:
        record = PATIENT_RECORDS.get(patient_id)
        if not record:
            return f"Sorry, no record found for patient ID {patient_id}."
        return (
            f"Patient Name: {record['name']} (ID: {patient_id})\n"
            f"Known conditions: {', '.join(record['conditions'])}\n"
            f"Medications: {', '.join(record['medications'])}"
        )
