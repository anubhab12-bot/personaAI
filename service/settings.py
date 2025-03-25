import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

# Rate limits
MAX_REQUESTS_PER_DAY = 1000
MAX_TOKENS_PER_DAY = 100000
MAX_TOKENS_PER_MINUTE = 6000
