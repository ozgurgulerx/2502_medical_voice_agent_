"""
Main entry point for the medical assistant application.
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required Azure OpenAI environment variables
required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set these variables in your .env file.")
    print("Example .env file:")
    print("AZURE_OPENAI_API_KEY=your_api_key_here")
    print("AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com")
    exit(1)

# Import the MedicalAssistant class
from medical_assistant import MedicalAssistant

async def main():
    """
    Initialize and run the medical assistant.
    """
    try:
        # Initialize the medical assistant
        print("Initializing Medical Assistant...")
        assistant = MedicalAssistant()
        
        # Start the conversation
        print("Starting conversation...")
        await assistant.start_conversation()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Medical Assistant session ended.")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())