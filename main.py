import time
import json
from prompts_folder.prompts import *
from service.settings import (
    MAX_REQUESTS_PER_DAY, 
    MAX_TOKENS_PER_DAY, 
    MAX_TOKENS_PER_MINUTE
)
from service.personal_knowledge import PersonalKnowledgeBase
from service.chat_service import ChatService

json_file = 'jason_data/personal_data.json'

def read_personal_data():
    """Read personal data from JSON file"""
    try:
        with open('jason_data/personal_data.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: personal_data.json not found in data directory")
        return {}
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in personal_data.json")
        return {}

def chatbot():
    """Main chatbot function."""
    # print(WELCOME_TEMPLATE)
    
    # Initialize components
    personal_kb = PersonalKnowledgeBase()
    personal_data = read_personal_data()
    personal_kb.add_personal_data(personal_data)
    chat_service = ChatService(personal_kb)
    print("Personal knowledge base initialized")
    # Initialize tracking variables
    total_requests = 0
    total_used_tokens = 0
    minute_start_time = time.time()
    tokens_used_in_minute = 0

    while True:
        # print(json_file)
        user_query = input("\nAsk me anything (or type 'quit' to quit): ")

        if user_query.lower() == "quit":
            print("\nGoodbye!")
            break


        if total_requests >= MAX_REQUESTS_PER_DAY:
            print("Daily request limit reached. Try again tomorrow.")
            break
        if total_used_tokens >= MAX_TOKENS_PER_DAY:
            print("Daily token limit reached. Try again tomorrow.")
            break

        if time.time() - minute_start_time >= 60:
            tokens_used_in_minute = 0
            minute_start_time = time.time()

        if tokens_used_in_minute >= MAX_TOKENS_PER_MINUTE:
            print("Token limit per minute reached. Waiting before next request...")
            time.sleep(60 - (time.time() - minute_start_time))
            tokens_used_in_minute = 0

        response_data, used_tokens = chat_service.chat(user_query)

        total_requests += 1
        total_used_tokens += used_tokens
        tokens_used_in_minute += used_tokens

        # Print Response
        print("\nAI Response:")
        print(response_data["response"])

        print("\nRelevant Links:")
        for link in response_data["links"]:
            print(f"{link['title']}: {link['url']}")

        print(f"\nTokens Used: {used_tokens}")
        print(f"Remaining Tokens Today: {MAX_TOKENS_PER_DAY - total_used_tokens}")
        print(f"Remaining Requests Today: {MAX_REQUESTS_PER_DAY - total_requests}")

if __name__ == "__main__":
    chatbot()