from enum import Enum
import json
import numpy as np
from groq import Groq
from .settings import GROQ_API_KEY, GROQ_MODEL
from .personal_knowledge import PersonalKnowledgeBase
from prompts_folder.prompts import *
from .utils import fetch_duckduckgo_links
from sklearn.metrics.pairwise import cosine_similarity

class QueryIntent(Enum):
    COMPANY = "company"
    IDENTITY = "identity"
    GENERAL = "general"

class ChatService:
    def __init__(self, personal_kb: PersonalKnowledgeBase):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.personal_kb = personal_kb
        
        # Initialize intent embeddings
        self.intent_embeddings = self._initialize_intent_embeddings()
        
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_TEMPLATE
            }
        ]

    def _initialize_intent_embeddings(self):
        """Initialize embeddings for different query intents"""
        intent_examples = {
            QueryIntent.COMPANY: [
                "tell me about the company",
                "what company do you work for",
                "company information",
                "workplace details",
                "employer information"
            ],
            QueryIntent.IDENTITY: [
                "who am i",
                "tell me about myself",
                "what's my background",
                "personal information",
                "my profile"
            ],
            # QueryIntent.SKILLS: [
            #     "what are my skills",
            #     "my abilities",
            #     "my technical skills",
            #     "what can i do",
            #     "my expertise"
            # ]
        }
        
        intent_vectors = {}
        for intent, examples in intent_examples.items():
            example_embeddings = [
                self.personal_kb.model.encode(ex) 
                for ex in examples
            ]
            intent_vectors[intent] = np.mean(example_embeddings, axis=0)
            
        return intent_vectors

    def detect_intent(self, query: str) -> QueryIntent:
        """Detect the intent of a query using semantic similarity"""
        query_embedding = self.personal_kb.model.encode(query)
        
        similarities = {}
        for intent, intent_embedding in self.intent_embeddings.items():
            similarity = cosine_similarity(
                [query_embedding], 
                [intent_embedding]
            )[0][0]
            similarities[intent] = similarity
        
        max_intent = max(similarities.items(), key=lambda x: x[1])
        
        if max_intent[1] < 0.3:
            return QueryIntent.GENERAL
            
        return max_intent[0]

    def format_template(self, template: str, **kwargs) -> str:
        """Format a template with the given kwargs"""
        return template.format(**kwargs)

    def handle_company_query(self, query: str):
        """Handle company-related queries"""
        try:
            # Get company-specific context
            context = self.personal_kb.get_relevant_context("company", top_k=5)
            
            # Format the prompt
            formatted_prompt = self.format_template(
                COMPANY_QUERY_TEMPLATE,
                personal_context=context,
                query=query
            )
            
            # Get response
            self.messages.append({"role": "user", "content": formatted_prompt})
            
            chat_response = self.client.chat.completions.create(
                messages=self.messages,
                model=GROQ_MODEL,
            )

            response_text = chat_response.choices[0].message.content
            
            # Attempt to parse the JSON response
            try:
                res_json = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                res_json = {
                    "response": "I apologize, but I encountered an error while processing the response.",
                    "links": []
                }
            
            # Retrieve company name from the knowledge base
            company_name = self.personal_kb.get_value('company', 'your company')
            
            # Fetch company website using DuckDuckGo
            search_query = f"{company_name} official site"
            links = fetch_duckduckgo_links(search_query)
            
            # Add the first link as the company website if available
            if links:
                res_json["links"] = [{
                    "title": f"{company_name} Website",
                    "url": links[0]['url']
                }]
            else:
                res_json["links"] = []
            
            self.messages.append({"role": "assistant", "content": json.dumps(res_json)})
            
            return res_json, chat_response.usage.total_tokens
        
        except Exception as e:
            # Log the exception for debugging
            print(f"Error retrieving company information: {e}")
            
            # Fallback response in case of errors
            error_response = {
                "response": "I apologize, but I encountered an error while retrieving company information.",
                "links": []
            }
        return error_response, 0
    
    def handle_identity_query(self, query: str):
        """Handle identity-related queries"""
        context = self.personal_kb.get_relevant_context(query, top_k=5)
        
        formatted_prompt = self.format_template(
            IDENTITY_QUERY_TEMPLATE,
            personal_context=context,
            query=query
        )
        
        return self._get_chat_response(formatted_prompt)

    def handle_identity_query(self, query: str):
        """Handle identity-related queries"""
        context = self.personal_kb.get_relevant_context(query, top_k=5)
        
        formatted_prompt = self.format_template(
            IDENTITY_QUERY_TEMPLATE,
            personal_context=context,
            query=query
        )
        
        return self._get_chat_response(formatted_prompt)

    # def handle_skills_query(self, query: str):
    #     """Handle skills-related queries"""
    #     context = self.personal_kb.get_relevant_context("skills expertise abilities")
        
    #     formatted_prompt = self.format_template(
    #         SKILLS_QUERY_TEMPLATE,
    #         personal_context=context,
    #         query=query
    #     )
        
    #     return self._get_chat_response(formatted_prompt)

    def handle_general_query(self, query: str):
        """Handle general queries"""
        context = self.personal_kb.get_relevant_context(query)
        
        formatted_prompt = self.format_template(
            GENERAL_QUERY_TEMPLATE,
            personal_context=context,
            query=query
        )
        
        response, tokens = self._get_chat_response(formatted_prompt)
        
        # Add DuckDuckGo links for general queries
        links = fetch_duckduckgo_links(query)
        response["links"] = links
        
        return response, tokens

    def _get_chat_response(self, prompt: str):
        """Helper method to get chat response"""
        self.messages.append({"role": "user", "content": prompt})
        
        chat_response = self.client.chat.completions.create(
            messages=self.messages,
            model=GROQ_MODEL,
        )
        
        response_text = chat_response.choices[0].message.content
        res_json = json.loads(response_text)
        
        self.messages.append({"role": "assistant", "content": json.dumps(res_json)})
        
        return res_json, chat_response.usage.total_tokens

    def chat(self, query: str):
        """Main chat function with intent-based routing"""
        intent = self.detect_intent(query)
        
        if intent == QueryIntent.COMPANY:
            return self.handle_company_query(query)
        elif intent == QueryIntent.IDENTITY:
            return self.handle_identity_query(query)
        else:
            return self.handle_general_query(query)