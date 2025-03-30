import time
import json
import os
import traceback
from prompts_folder.prompts import *
from service.send_mail import is_valid_email, send_email
from service.settings import (
    MAX_REQUESTS_PER_DAY, 
    MAX_TOKENS_PER_DAY, 
    MAX_TOKENS_PER_MINUTE
)
from service.personal_knowledge import PersonalKnowledgeBase
from service.chat_service import ChatService
from service.tts import TextToSpeech
import webbrowser
import re
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def is_negative_response(response_text):
    labels = ["negative response", "positive response"]
    result = classifier(response_text, labels)
    return result["labels"][0] == "negative response"

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
    print("Initializing chatbot with LangChain and Hugging Face embeddings...")
    
    # Create data directory for saving learned examples
    os.makedirs("data", exist_ok=True)
    
    try:
        # Initialize components
        personal_kb = PersonalKnowledgeBase()
        personal_data = read_personal_data()
        personal_kb.add_personal_data(personal_data)
        
        chat_service = ChatService(personal_kb)
        print("Personal knowledge base initialized")
        
        tts = TextToSpeech()
        
        # Initialize tracking variables
        total_requests = 0
        total_used_tokens = 0
        minute_start_time = time.time()
        tokens_used_in_minute = 0

        last_response_text = ""

        while True:
            user_query = input("\nAsk me anything (or type 'quit' to quit): ")

            token_count = chat_service.count_tokens(user_query)

            if token_count > chat_service.max_tokens_per_query:
                response = {
                    "response": f"Your query is too long. Please limit your input to {chat_service.max_tokens_per_query} tokens.",
                    "links": []
                }
                return response, 0
                

            topic = chat_service.detect_topic(user_query)
            chat_service.topics.append(topic)

            intent = chat_service.detect_intent(user_query)
            chat_service.intents.append(intent)

            if user_query.lower() == "quit":
                print("\nSaving learned data and exiting...")
                chat_service.save_intent_examples()
                chat_service.save_conversation_history()
                print("Goodbye!")
                break

            if "voice" in user_query.lower():
                user_query = input("\n Do you want 'female' or 'male' voice :")
                if "female" in user_query.lower():
                    if last_response_text:  # Check if there is a last response to speak
                        tts.speak_female(last_response_text)  # Convert last response to speech
                elif "male" in user_query.lower():
                    if last_response_text:  # Check if there is a last response to speak
                        tts.speak_male(last_response_text)
                else:
                    print("No previous response to convert to voice.")
                continue  # Skip the rest of the loop
            

            if "send" and "email" in user_query.lower():
                user_query = input("\n Do you want to send a mail ?")
                if "yes" in user_query.lower():
                        recipient_email = input("\nPlease add the email of the person: ")
                        if not is_valid_email(recipient_email):
                            return "Error: The email address provided is not valid. Please enter a valid email address."
                        subject = input("\nEnter the subject of the email: ")
                        body = last_response_text

                        document_path = input("\nPlease provide the path to the PDF document you want to attach: ")
                        document_path = document_path.strip()  # Remove any leading/trailing whitespace

                        # Check if the document path is valid
                        if document_path and not os.path.isfile(document_path):
                            document_path = None
                            print("Error: The provided document path is not valid or the file does not exist.")
                            continue
                        send_email( subject, body, recipient_email, document_path)
                        continue 
                
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

            url_pattern =  r"\(?(https://[^\s\)]+)\)?"
            
            response_data, used_tokens = chat_service.chat(user_query)
            response_data["Tasks_doc"] = chat_service.save_topic_wise_conversation_history()
            last_response_text = response_data["response"]
            print(last_response_text)

            total_requests += 1
            total_used_tokens += used_tokens
            tokens_used_in_minute += used_tokens

            urls = re.findall(url_pattern, response_data["response"])

            for url in urls:
                print(f"Opening: {url}")
                webbrowser.open(url)

            print("\nPersonaAI:")
            print("Response : ", response_data["response"])
            print("----------------------------------------------------------")
            print("File path : ", response_data["Tasks_doc"])



            if response_data.get("links") and len(response_data["links"]) > 0:
                print("\nRelevant Links:")
                for link in response_data["links"]:
                    if "title" in link and "url" in link:
                        print(f"• {link['title']}: {link['url']}")
                    elif "url" in link:
                        print(f"• {link['url']}")

            # return response_data["response"]
            # Print usage stats in a more subtle way
            print(f"\n[Tokens: {used_tokens} | Remaining: {MAX_TOKENS_PER_DAY - total_used_tokens}]")
            
    except KeyboardInterrupt:
        print("\nProgram interrupted. Saving learned data...")
        chat_service.save_intent_examples()
        print("Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print(traceback.format_exc())
        print("Attempting to save learned data before exiting...")
        try:
            chat_service.save_intent_examples()
            print("Learned data saved successfully.")
        except:
            print("Could not save learned data.")

if __name__ == "__main__":
    chatbot()