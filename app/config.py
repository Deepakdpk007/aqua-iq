import os
from dotenv import load_dotenv

load_dotenv()

AZURE_API_KEY = os.getenv("AZUREOPENAIAPI_KEY")
AZURE_ENDPOINT = os.getenv("AZUREOPENAI_ENDPOINT")
