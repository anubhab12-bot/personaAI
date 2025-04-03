import time
import json
import os
import traceback
from prompts_folder.prompts import *
from service.check import EnhancedGoogleSearch
from service.send_mail import is_valid_email, send_email
from service.settings import (
    MAX_REQUESTS_PER_DAY, 
    MAX_TOKENS_PER_DAY, 
    MAX_TOKENS_PER_MINUTE
)
from service.personal_knowledge import PersonalKnowledgeBase
from service.chat_service import ChatService
from service.taskautomation import  process_with_groq, search_google
from service.tts import TextToSpeech
import webbrowser
import re
from transformers import pipeline
from langchain_community.utilities import SearxSearchWrapper
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import webbrowser
import re

from service.utils import fetch_duckduckgo_links

s = SearxSearchWrapper(searx_host="http://localhost:8888")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

search_tool = EnhancedGoogleSearch()

def is_website_request(query: str, llm) -> bool:
    """Use LLM to determine if the user wants to open a website"""
    system_prompt = """You are an intent classifier for website navigation requests.
    Determine if the user is asking to open, visit, or navigate to a website.
    Consider both explicit requests ("open google.com") and implicit ones ("I want to check my facebook").
    
    Respond with ONLY "true" if it's a website navigation request, or "false" if it's not.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    try:
        response = llm.invoke(messages).content.strip().lower()
        return response == "true"
    except Exception as e:
        print(f"Error in website intent classification: {e}")
        return False

def extract_and_open_website(query: str, llm) -> str:
    """Extract and open website using LLM understanding"""
    system_prompt = """You are an AI assistant helping with website navigation.
    Given the user's request, determine the appropriate website URL.
    
    Guidelines:
    1. For explicit domains (e.g., "google.com"), add https://www. if needed
    2. For service names (e.g., "google"), infer the most likely domain
    3. For intent-based requests (e.g., "I want to watch videos"), suggest the most appropriate platform
    4. For ambiguous requests, choose the most popular/relevant option
    
    Return ONLY the complete URL (e.g., "https://www.google.com") without any additional text.
    If no valid website can be determined, return "INVALID_REQUEST"
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    try:
        response = llm.invoke(messages).content.strip()
        
        if response == "INVALID_REQUEST":
            return "I couldn't determine which website you want to visit. Could you please be more specific?"
            
        if response.startswith("http"):
            webbrowser.open(response)
            return f"I've opened {response} for you."
        else:
            return "I couldn't process the website request properly."
    except Exception as e:
        print(f"Error processing website request: {e}")
        return "Sorry, I encountered an error trying to open the website."
    
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

            if is_website_request(user_query, chat_service.llm):
                response_text = extract_and_open_website(user_query, chat_service.llm)
                print(f"🌐 {response_text}")
                last_response_text = response_text
                
                # Update conversation history
                chat_service.chat_history.append(HumanMessage(content=user_query))
                chat_service.chat_history.append(AIMessage(content=response_text))
                continue

            token_count = chat_service.count_tokens(user_query)

            if token_count > chat_service.max_tokens_per_query:
                response = {
                    "response": f"⚠️ Your query is too long. Please limit your input to {chat_service.max_tokens_per_query} tokens.",
                    "links": []
                }
                return response, 0
                

            topic = chat_service.detect_topic(user_query)
            chat_service.topics.append(topic)

            intent = chat_service.detect_intent(user_query)
            chat_service.intents.append(intent)

            if user_query.lower() == "quit":
                print("\n💾 Saving learned data and exiting...")
                chat_service.save_intent_examples()
                chat_service.save_conversation_history()
                print("👋 Goodbye!")
                break

            if "voice" in user_query.lower():
                user_query = input("\n🎤 Do you want 'female' or 'male' voice :")
                if "female" in user_query.lower():
                    if last_response_text:  # Check if there is a last response to speak
                        tts.speak_female(last_response_text)  # Convert last response to speech
                elif "male" in user_query.lower():
                    if last_response_text:  # Check if there is a last response to speak
                        tts.speak_male(last_response_text)
                else:
                    print("❌ No previous response to convert to voice.")
                continue  # Skip the rest of the loop
            

            if "send" and "email" in user_query.lower():
                user_query = input("\n📧 Do you want to send a mail ?")
                if "yes" in user_query.lower():
                        recipient_email = input("\n📧 Please add the email of the person: ")
                        if not is_valid_email(recipient_email):
                            return "❌ Error: The email address provided is not valid. Please enter a valid email address."
                        subject = input("\n📝 Enter the subject of the email: ")
                        banner = input("🎨 Add a custom banner : ")
                        body = last_response_text

                        document_path = input("\n📄 Please provide the path to the PDF document you want to attach: ")
                        document_path = document_path.strip()  # Remove any leading/trailing whitespace

                        # Check if the document path is valid
                        if document_path and not os.path.isfile(document_path):
                            document_path = None
                            print("❌ Error: The provided document path is not valid or the file does not exist.")
                            continue
                        send_email(subject, body, recipient_email, document_path, banner)
                        print("✅ Email sent successfully!")
                        continue 
                
            if total_requests >= MAX_REQUESTS_PER_DAY:
                print("⚠️ Daily request limit reached. Try again tomorrow.")
                break
            if total_used_tokens >= MAX_TOKENS_PER_DAY:
                print("⚠️ Daily token limit reached. Try again tomorrow.")
                break

            if time.time() - minute_start_time >= 60:
                tokens_used_in_minute = 0
                minute_start_time = time.time()

            if tokens_used_in_minute >= MAX_TOKENS_PER_MINUTE:
                print("⏳ Token limit per minute reached. Waiting before next request...")
                time.sleep(60 - (time.time() - minute_start_time))
                tokens_used_in_minute = 0

            url_pattern =  r"\(?(https://[^\s\)]+)\)?"
            
            response_data, used_tokens = chat_service.chat(user_query)
            response_data["Tasks_doc"] = chat_service.save_topic_wise_conversation_history()
            last_response_text = response_data["response"]
            
            # Map intents to emojis
            intent_emojis = {
                "greeting": "👋",
                "question": "❓",
                "information": "ℹ️",
                "help": "🆘",
                "confirmation": "✅",
                "error": "❌",
                "warning": "⚠️",
                "success": "🎉",
                "thinking": "🤔",
                "suggestion": "💡",
                "code": "💻",
                "link": "🔗",
                "email": "📧",
                "file": "📄",
                "search": "🔍",
                "voice": "🎤",
                "website": "🌐",
                "document": "📁",
                "data": "📊",
                "automation": "⚙️",
                "personal": "👤",
                "company": "🏢",
                "identity": "🎭",
                "knowledge": "📚",
                "positive": "😊",
                "negative": "😕",
                "apology": "🙏",
                "explanation": "📝",
                "summary": "📋",
                "default": "🤖"
            }

            # Map emotions to emojis
            emotion_emojis = {
                "excitement": "🎉",
                "happiness": "😊",
                "celebration": "🎊",
                "sports": "⚽",
                "cricket": "🏏",
                "victory": "🏆",
                "party": "🎈",
                "birthday": "🎂",
                "wedding": "💍",
                "customization": "🎨",
                "update": "🔄",
                "feature": "✨",
                "announcement": "📢",
                "news": "📰",
                "thrilling": "🎢",
                "crowd": "👥",
                "fun": "🎮",
                "share": "📤",
                "special": "🌟",
                "default": "🤖"
            }

            # Use LLM to detect emotions in the response
            emotion_prompt = """Analyze the following text and identify the main emotions and themes present.
            Return a comma-separated list of emotions/themes from this list: excitement, happiness, celebration, 
            sports, cricket, victory, party, birthday, wedding, customization, update, feature, announcement, 
            news, thrilling, crowd, fun, share, special.
            
            Text: {text}
            
            Return ONLY the comma-separated list, nothing else."""
            
            try:
                emotion_messages = [
                    SystemMessage(content=emotion_prompt),
                    HumanMessage(content=last_response_text)
                ]
                emotion_response = chat_service.llm.invoke(emotion_messages).content.strip()
                detected_emotions = [e.strip() for e in emotion_response.split(',')]
                
                # Get emojis for detected emotions
                emotion_emojis_list = []
                for emotion in detected_emotions:
                    if emotion in emotion_emojis:
                        emotion_emojis_list.append(emotion_emojis[emotion])
                
                # Combine intent and emotion emojis
                if emotion_emojis_list:
                    response_emoji = " ".join(emotion_emojis_list)
                else:
                    # Get emoji based on intent as fallback
                    intent_str = str(intent).lower()
                    response_emoji = intent_emojis.get(intent_str, intent_emojis["default"])
                
            except Exception as e:
                # Fallback to intent-based emoji if emotion detection fails
                intent_str = str(intent).lower()
                response_emoji = intent_emojis.get(intent_str, intent_emojis["default"])
            
            # Special case for negative responses
            if is_negative_response(last_response_text):
                response_emoji = "😕"
            
            # Special case for website responses
            if "http" in last_response_text or "www." in last_response_text:
                response_emoji = "🔗"

            print(f"\n{response_emoji} PersonaAI:")
            print(f"{response_emoji} Response : ", response_data["response"])
            print("📂 File path : ", response_data["Tasks_doc"])

            if response_data.get("links") and len(response_data["links"]) > 0:
                print("\n🔗 Relevant Links:")
                for link in response_data["links"]:
                    if "title" in link and "url" in link:
                        print(f"• {link['title']}: {link['url']}")
                    elif "url" in link:
                        print(f"• {link['url']}")

            # Print usage stats with emoji
            print(f"\n📊 [Tokens: {used_tokens} | Remaining: {MAX_TOKENS_PER_DAY - total_used_tokens}]")
            
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