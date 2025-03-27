from enum import Enum
import json
import numpy as np
import os
import pickle
from groq import Groq
from typing import Dict, List, Tuple, Any, Optional
from .settings import GROQ_API_KEY, GROQ_MODEL
from .personal_knowledge import PersonalKnowledgeBase
from prompts_folder.prompts import *
from .utils import fetch_duckduckgo_links

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MultiQueryRetriever,
    TimeWeightedVectorStoreRetriever,
    BM25Retriever,
    EnsembleRetriever
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter
)
from langchain.chains import (
    LLMChain,
    RetrievalQA,
    ConversationalRetrievalChain,
    SequentialChain
)
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.schema import BaseRetriever

class QueryIntent(Enum):
    COMPANY = "company"
    IDENTITY = "identity"
    GENERAL = "general"

class ChatService:
    def __init__(self, personal_kb: PersonalKnowledgeBase):
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL
        )
        
        self.personal_kb = personal_kb
        
        # Initialize LangChain HuggingFace embeddings with a strong model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",  # More accurate model
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better matching
        )
        
        # Initialize various LangChain components
        self._initialize_vector_stores()
        self._initialize_retrievers()
        self._initialize_intent_classification()
        self._initialize_chains()
        
        # Track conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.chat_history = []
        
        # Initialize tracking for dynamic learning
        self.intent_examples, self.intent_centroids = self._initialize_intent_examples()
        self.classified_queries = {intent: [] for intent in QueryIntent}

    def _initialize_vector_stores(self):
        """Initialize different vector stores for different types of knowledge"""
        documents = self._prepare_documents_from_kb()
        
        # Use recursive text splitter for better semantic chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # Create main vector store with FAISS
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        # Create separate stores for different categories
        self.company_docs = [d for d in split_docs if d.metadata.get("category") == "company"]
        self.identity_docs = [d for d in split_docs if d.metadata.get("category") == "identity"]
        
        # Create specialized vector stores if we have enough documents
        if self.company_docs:
            self.company_store = FAISS.from_documents(self.company_docs, self.embeddings)
        else:
            self.company_store = self.vector_store
            
        if self.identity_docs:
            self.identity_store = FAISS.from_documents(self.identity_docs, self.embeddings)
        else:
            self.identity_store = self.vector_store
        
        # Optional: Create a BM25 (keyword-based) store for hybrid search
        try:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 5
        except:
            self.bm25_retriever = None
            print("BM25Retriever could not be initialized, using vector search only")

    def _prepare_documents_from_kb(self) -> List[Document]:
        """Convert personal knowledge to LangChain documents with metadata"""
        documents = []
        
        # Process structured knowledge
        for category, content in self.personal_kb.knowledge.items():
            if isinstance(content, list):
                for i, item in enumerate(content):
                    documents.append(Document(
                        page_content=item,
                        metadata={
                            "category": category, 
                            "index": i,
                            "source": "knowledge_base",
                            "time_created": 0  # For time-weighted retrieval
                        }
                    ))
            elif isinstance(content, dict):
                for subcategory, text in content.items():
                    documents.append(Document(
                        page_content=str(text),
                        metadata={
                            "category": category, 
                            "subcategory": subcategory,
                            "source": "knowledge_base",
                            "time_created": 0
                        }
                    ))
            else:
                documents.append(Document(
                    page_content=str(content),
                    metadata={
                        "category": category,
                        "source": "knowledge_base",
                        "time_created": 0
                    }
                ))
                
        return documents

    def _initialize_retrievers(self):
        """Initialize various retrievers for different purposes"""
        # Basic retrievers
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        self.company_retriever = self.company_store.as_retriever(
            search_kwargs={"k": 3}
        )
        
        self.identity_retriever = self.identity_store.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Advanced: Create a time-weighted retriever that can prioritize recent info
        self.time_weighted_retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=self.vector_store,
            decay_rate=0.01,
            k=5
        )
        
        # Multi-query retriever to generate variations of the user's query
        try:
            self.multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=self.retriever,
                llm=self.llm
            )
        except Exception as e:
            print(f"Could not initialize MultiQueryRetriever: {e}")
            self.multi_query_retriever = self.retriever
        
        # Create ensemble retriever if BM25 is available
        if self.bm25_retriever:
            try:
                self.hybrid_retriever = EnsembleRetriever(
                    retrievers=[self.retriever, self.bm25_retriever],
                    weights=[0.7, 0.3]  # Prioritize vector retriever
                )
            except Exception as e:
                print(f"Could not initialize EnsembleRetriever: {e}")
                self.hybrid_retriever = self.retriever
        else:
            self.hybrid_retriever = self.retriever
        
        # Compression retriever to filter irrelevant results
        try:
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.7
            )
            
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=self.hybrid_retriever
            )
        except Exception as e:
            print(f"Could not initialize ContextualCompressionRetriever: {e}")
            self.compression_retriever = self.hybrid_retriever

    def _initialize_intent_classification(self):
        """Initialize LangChain components for intent classification"""
        # Create an LLM chain for explicit intent classification
        intent_template = """You are an intent classifier.
        Classify the following query into exactly one of these categories:
        - company: queries about the company, workplace, or employer
        - identity: queries about the user's identity, background, profile
        - general: any other type of query
        
        Query: {query}
        
        Format your response as a single word: "company", "identity", or "general".
        """
        
        intent_prompt = ChatPromptTemplate.from_template(intent_template)
        
        try:
            self.intent_classifier = (
                intent_prompt | 
                self.llm | 
                StrOutputParser()
            )
        except Exception as e:
            print(f"Could not initialize intent classifier: {e}")
            self.intent_classifier = None

    def _initialize_intent_examples(self):
        """Initialize intent examples with dynamic learning capability"""
        # Path to saved intent examples
        examples_path = "data/intent_examples.pkl"
        
        # Starting examples for each intent with more comprehensive examples
        intent_examples = {
            QueryIntent.COMPANY: [
                "tell me about the company",
                "what company do you work for",
                "company information",
                "workplace details",
                "employer information",
                "company culture",
                "company mission",
                "what products does the company make",
                "who is the CEO of the company",
                "company revenue"
            ],
            QueryIntent.IDENTITY: [
                "who am i",
                "tell me about myself",
                "what's my background",
                "personal information",
                "my profile",
                "my education",
                "my experience",
                "what skills do I have",
                "my professional achievements",
                "where did I go to school"
            ],
        }
        
        # Try to load previously saved intent examples
        if os.path.exists(examples_path):
            try:
                with open(examples_path, "rb") as f:
                    saved_examples = pickle.load(f)
                    # Merge saved examples with default ones
                    for intent, examples in saved_examples.items():
                        if intent in intent_examples:
                            # Use set to avoid duplicates, then convert back to list
                            intent_examples[intent] = list(set(intent_examples[intent] + examples))
                print(f"Loaded {sum(len(examples) for examples in saved_examples.values())} learned intent examples")
            except (pickle.PickleError, EOFError) as e:
                print(f"Error loading saved intent examples: {e}")
        
        # Calculate initial centroids
        intent_centroids = self._calculate_centroids(intent_examples)
        
        return intent_examples, intent_centroids

    def _calculate_centroids(self, intent_examples):
        """Calculate centroids from examples using our embeddings model"""
        intent_centroids = {}
        for intent, examples in intent_examples.items():
            # Batch encode all examples for efficiency
            example_embeddings = [
                self.embeddings.embed_query(ex) 
                for ex in examples
            ]
            intent_centroids[intent] = np.mean(example_embeddings, axis=0)
        return intent_centroids

    def _initialize_chains(self):
        """Initialize LangChain chains for different query types"""
        try:
            # Create a conversational QA chain for company queries
            self.company_chain = self._create_qa_chain(
                self.company_retriever,
                "You are an AI assistant providing information about the company. Answer based on the context provided."
            )
            
            # Create a conversational QA chain for identity queries
            self.identity_chain = self._create_qa_chain(
                self.identity_retriever,
                "You are an AI assistant providing information about the user. Answer based on the context provided."
            )
            
            # Create a general QA chain
            self.general_chain = self._create_qa_chain(
                self.compression_retriever,
                "You are a helpful AI assistant. Answer based on the context when relevant, or acknowledge when you don't know."
            )
        except Exception as e:
            print(f"Error initializing chains: {e}")
            # Fallback to direct LLM if chains fail
            self.company_chain = None
            self.identity_chain = None
            self.general_chain = None

    def _create_qa_chain(self, retriever, system_message):
        """Create a conversational QA chain with the given retriever and system message"""
        # Define the prompt for the chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Create a function to format documents
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        
        # Create the chain
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": RunnableLambda(lambda _: self.chat_history)}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain

    def save_intent_examples(self):
        """Save learned intent examples for future use"""
        os.makedirs("data", exist_ok=True)
        try:
            with open("data/intent_examples.pkl", "wb") as f:
                pickle.dump(self.intent_examples, f)
            print(f"Saved {sum(len(examples) for examples in self.intent_examples.values())} intent examples")
        except Exception as e:
            print(f"Error saving intent examples: {e}")

    def detect_intent(self, query: str) -> QueryIntent:
        """Detect the intent of a query with special handling for identity questions"""
        from prompts_folder.prompts import is_assistant_identity_question
        
        # If it's asking about the assistant's name/identity, don't confuse with user identity
        if is_assistant_identity_question(query):
            # Skip intent detection for assistant identity questions - handled separately
            return QueryIntent.GENERAL
        
        # First, try embedding-based classification
        query_embedding = self.embeddings.embed_query(query)
        
        similarities = {}
        for intent, intent_embedding in self.intent_centroids.items():
            similarity = np.dot(query_embedding, intent_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(intent_embedding)
            )
            similarities[intent] = similarity
        
        max_intent = max(similarities.items(), key=lambda x: x[1])
        embedding_intent = max_intent[0] if max_intent[1] >= 0.3 else QueryIntent.GENERAL
        embedding_confidence = max_intent[1]
        
        # If confidence is low and LLM classifier is available, use it
        if embedding_confidence < 0.5 and self.intent_classifier:
            try:
                llm_result = self.intent_classifier.invoke({"query": query})
                llm_result = llm_result.lower().strip()
                
                if "company" in llm_result:
                    llm_intent = QueryIntent.COMPANY
                elif "identity" in llm_result:
                    llm_intent = QueryIntent.IDENTITY
                else:
                    llm_intent = QueryIntent.GENERAL
                    
                # Use LLM result when embedding confidence is low
                final_intent = llm_intent
            except Exception as e:
                print(f"Error in LLM intent classification: {e}")
                final_intent = embedding_intent
        else:
            # Use embedding result when confidence is high or LLM not available
            final_intent = embedding_intent
        
        # Store this query with its classification for potential retraining
        self.classified_queries[final_intent].append(query)
        
        # Dynamically update the model if confidence is high
        if embedding_confidence > 0.6:  # High confidence threshold
            self.intent_examples[final_intent].append(query)
            # Limit the size of examples to prevent it from growing too large
            max_examples = 50
            if len(self.intent_examples[final_intent]) > max_examples:
                self.intent_examples[final_intent] = self.intent_examples[final_intent][-max_examples:]
            
            # Periodically update centroids
            if len(self.classified_queries[final_intent]) % 5 == 0:
                self.intent_centroids = self._calculate_centroids(self.intent_examples)
        
        return final_intent

    def handle_company_query(self, query: str) -> Tuple[Dict[str, Any], int]:
        """Handle company-related queries"""
        try:
            if self.company_chain:
                # Use the company chain to get a response
                response_text = self.company_chain.invoke(query)
            else:
                # Fallback to direct context retrieval
                context = self.get_relevant_context(query, category="company", top_k=5)
                response_text = self._get_direct_response(query, context, "company information")
            
            # Create JSON response
            response = {
                "response": response_text,
                "links": []
            }
            
            # Retrieve company name from the knowledge base
            company_name = self.personal_kb.get_value('company', 'your company')
            
            # Fetch company website using DuckDuckGo
            search_query = f"{company_name} official site"
            links = fetch_duckduckgo_links(search_query)
            
            # Add the first link as the company website if available
            if links:
                response["links"] = [{
                    "title": f"{company_name} Website",
                    "url": links[0]['url']
                }]
            
            # Update conversation history
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response_text))
            
            # Store in memory
            self.memory.save_context({"input": query}, {"output": response_text})
            
            # Estimate token usage (very approximate)
            tokens = len(query.split()) + len(response_text.split()) * 1.5
            
            return response, int(tokens)
        
        except Exception as e:
            print(f"Error retrieving company information: {e}")
            error_response = {
                "response": "I apologize, but I encountered an error while retrieving company information.",
                "links": []
            }
            return error_response, 0

    def handle_identity_query(self, query: str) -> Tuple[Dict[str, Any], int]:
        """Handle identity-related queries with special handling for assistant's name"""
        from prompts_folder.prompts import is_assistant_identity_question
        
        # If asking about the assistant, not the user
        if is_assistant_identity_question(query):
            response_text = "I am PersonaAI, your AI assistant. How can I help you today?"
            
            response = {
                "response": response_text,
                "links": []
            }
            
            # Update conversation history
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response_text))
            
            tokens = len(query.split()) + len(response_text.split()) * 1.5
            return response, int(tokens)
        
        # Otherwise process as a normal identity query about the user
        try:
            if self.identity_chain:
                # Use the identity chain to get a response
                response_text = self.identity_chain.invoke(query)
            else:
                # Fallback to direct context retrieval
                context = self.get_relevant_context(query, category="identity", top_k=5)
                response_text = self._get_direct_response(query, context, "personal information")
            
            # Create JSON response
            response = {
                "response": response_text,
                "links": []
            }
            
            # Update conversation history
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response_text))
            
            # Store in memory
            self.memory.save_context({"input": query}, {"output": response_text})
            
            # Estimate token usage (very approximate)
            tokens = len(query.split()) + len(response_text.split()) * 1.5
            
            return response, int(tokens)
        
        except Exception as e:
            print(f"Error retrieving identity information: {e}")
            error_response = {
                "response": "I apologize, but I encountered an error while retrieving your information.",
                "links": []
            }
            return error_response, 0
    
    def handle_general_query(self, query: str) -> Tuple[Dict[str, Any], int]:
        """Handle general queries with improved context retrieval and response generation"""
        try:
            # Check if it's a question about PersonaAI first
            from prompts_folder.prompts import is_assistant_identity_question
            if is_assistant_identity_question(query):
                # Return identity response instead
                response_text = "I am PersonaAI, your AI assistant. How can I help you today?"
                response = {
                    "response": response_text,
                    "links": []
                }
                
                # Update conversation history
                self.chat_history.append(HumanMessage(content=query))
                self.chat_history.append(AIMessage(content=response_text))
                tokens = len(query.split()) + len(response_text.split()) * 1.5
                return response, int(tokens)
            
            # Enhanced context retrieval using multiple strategies
            basic_context = self.get_relevant_context(query, top_k=3)
            
            # Try to get additional context with the multi-query retriever if available
            try:
                if hasattr(self, 'multi_query_retriever') and self.multi_query_retriever:
                    enhanced_docs = self.multi_query_retriever.get_relevant_documents(query)
                    enhanced_context = "\n\n".join([doc.page_content for doc in enhanced_docs])
                    # Combine contexts, prioritizing the enhanced retrieval
                    combined_context = enhanced_context + "\n\n" + basic_context
                else:
                    combined_context = basic_context
            except Exception as e:
                print(f"Error using multi-query retriever: {e}")
                combined_context = basic_context
            
            # Use the general chain if available
            if hasattr(self, 'general_chain') and self.general_chain:
                try:
                    # Use the general chain to get a response
                    response_text = self.general_chain.invoke(query)
                except Exception as e:
                    print(f"Error using general chain: {e}. Falling back to direct response.")
                    # Fallback to direct context retrieval
                    from prompts_folder.prompts import GENERAL_QUERY_TEMPLATE
                    response_text = self._get_direct_response(
                        query, 
                        GENERAL_QUERY_TEMPLATE.format(query=query, personal_context=combined_context), 
                        "general information"
                    )
            else:
                # Fallback to direct context retrieval
                response_text = self._get_direct_response(
                    query, 
                    GENERAL_QUERY_TEMPLATE.format(query=query, personal_context=combined_context), 
                    "general information"
                )
            
            # Check if it's asking for an email draft
            if 'email' in query.lower() and ('write' in query.lower() or 'draft' in query.lower() or 'create' in query.lower()):
                # Create a JSON response with no links for email drafts
                response = {
                    "response": response_text,
                    "links": []
                }
            else:
                # Create a JSON response with DuckDuckGo links for general queries
                response = {
                    "response": response_text,
                    "links": []
                }
                
                # Add DuckDuckGo links only for knowledge-seeking queries, not conversational ones
                is_knowledge_query = any(keyword in query.lower() for keyword in [
                    'what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain', 'define', 'describe'
                ])
                
                if is_knowledge_query and 'email' not in query.lower():
                    try:
                        links = fetch_duckduckgo_links(query)
                        response["links"] = links
                    except Exception as e:
                        print(f"Error fetching links: {e}")
                        response["links"] = []
            
            # Update conversation history
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response_text))
            
            # Store in memory
            self.memory.save_context({"input": query}, {"output": response_text})
            
            # Estimate token usage (very approximate)
            tokens = len(query.split()) + len(response_text.split()) * 1.5
            
            return response, int(tokens)
        
        except Exception as e:
            print(f"Error handling general query: {e}")
            error_response = {
                "response": "I apologize, but I encountered an error while processing your query. How else can I assist you today?",
                "links": []
            }
            return error_response, 0
    def _get_direct_response(self, query, context, topic_type):
        """Get a direct response from the LLM when chains fail"""
        prompt = f"""
        You are a helpful AI assistant providing {topic_type}.
        
        Context information:
        {context}
        
        User query: {query}
        
        Provide a helpful response based on the context provided. If the context doesn't contain relevant information, acknowledge that you don't know.
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.invoke(messages)
        print(response.content)
        return response.content
    
    def get_relevant_context(self, query, category=None, top_k=5):
        """Get relevant context using appropriate retriever"""
        try:
            if category == "company":
                docs = self.company_retriever.get_relevant_documents(query)
            elif category == "identity":
                docs = self.identity_retriever.get_relevant_documents(query)
            else:
                # Use the most advanced retriever we have for general queries
                docs = self.compression_retriever.get_relevant_documents(query)
            
            # Format retrieved documents
            context = "\n\n".join([f"{doc.page_content}" for doc in docs])
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "No relevant context found."

    def chat(self, query: str) -> Tuple[Dict[str, Any], int]:
        """Main chat function with intent-based routing and specialized handling"""
        from prompts_folder.prompts import (
            is_greeting, is_assistant_identity_question, ASSISTANT_IDENTITY_TEMPLATE
        )
        
        # Check for questions about the assistant's name/identity first
        if is_assistant_identity_question(query):
            try:
                # Use the specific template for assistant identity questions
                response_text = self._get_direct_response(
                    query, 
                    ASSISTANT_IDENTITY_TEMPLATE.format(query=query), 
                    "assistant identity"
                )
                
                # Force check to ensure the response contains "PersonaAI"
                if "PersonaAI" not in response_text:
                    response_text = "I am PersonaAI, your AI assistant. How can I help you today?"
                
                response = {
                    "response": response_text,
                    "links": []
                }
                
                # Update conversation history
                self.chat_history.append(HumanMessage(content=query))
                self.chat_history.append(AIMessage(content=response_text))
                
                tokens = len(query.split()) + len(response_text.split()) * 1.5
                return response, int(tokens)
            except Exception as e:
                print(f"Error handling assistant identity question: {e}")
                # Fall back to a hard-coded response to ensure correct identity
                response = {
                    "response": "I am PersonaAI, your AI assistant. How can I help you today?",
                    "links": []
                }
                self.chat_history.append(HumanMessage(content=query))
                self.chat_history.append(AIMessage(content=response["response"]))
                return response, 30
        
        # Handle follow-up "why" questions
        elif query.lower().strip() in ["why", "why?"] and self.chat_history:
            # Get the previous response for context
            previous_response = self.chat_history[-1].content if self.chat_history else ""
            
            try:
                # Generate a follow-up explanation
                follow_up_prompt = f"""
                The user previously asked something, and you provided this response:
                "{previous_response}"
                
                Now the user is asking "why?". Provide a natural follow-up explanation that elaborates on your previous 
                response in a conversational way. Maintain the same friendly, warm tone.
                
                If the previous conversation was about your identity as PersonaAI, emphasize that you are PersonaAI,
                an AI assistant created to help the user.
                """
                
                response_text = self._get_direct_response(
                    query, 
                    follow_up_prompt, 
                    "follow-up explanation"
                )
                
                response = {
                    "response": response_text,
                    "links": []
                }
                
                # Update conversation history
                self.chat_history.append(HumanMessage(content=query))
                self.chat_history.append(AIMessage(content=response_text))
                
                tokens = len(query.split()) + len(response_text.split()) * 1.5
                return response, int(tokens)
                
            except Exception as e:
                print(f"Error handling 'why' follow-up: {e}")
                # Fall back to normal processing
        
        # Check if the query is asking about user identity
        # elif is_identity_question(query) and not is_assistant_identity_question(query):
        #     # Get relevant personal context
        #     context = self.get_relevant_context(query, top_k=5)
            
        #     if "who am i" in query.lower():
        #         # Use specific template for "who am I" questions
        #         response_text = self._get_direct_response(
        #             query, 
        #             WHO_AM_I_TEMPLATE.format(query=query, personal_context=context), 
        #             "identity"
        #         )
        #     else:
        #         # Handle other identity questions
        #         return self.handle_identity_query(query)
            
            response = {
                "response": response_text,
                "links": []
            }
            
            # Update conversation history
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response_text))
            
            tokens = len(query.split()) + len(response_text.split()) * 1.5
            return response, int(tokens)
        
        # Check if the query is a greeting
        elif is_greeting(query):
            # Use a simplified greeting response
            try:
                # Use greeting template directly
                from prompts_folder.prompts import GREETING_TEMPLATE
                response_text = self._get_direct_response(
                    query, 
                    GREETING_TEMPLATE.format(query=query), 
                    "greeting"
                )
                
                response = {
                    "response": response_text,
                    "links": []  # No links for greetings
                }
                
                # Update conversation history
                self.chat_history.append(HumanMessage(content=query))
                self.chat_history.append(AIMessage(content=response_text))
                
                tokens = len(query.split()) + len(response_text.split()) * 1.5
                return response, int(tokens)
            except Exception as e:
                print(f"Error handling greeting: {e}")
                # Fall back to normal intent-based routing

        # Proceed with normal intent-based routing for other queries
        intent = self.detect_intent(query)
        
        if intent == QueryIntent.COMPANY:
            return self.handle_company_query(query)
        elif intent == QueryIntent.IDENTITY:
            return self.handle_identity_query(query)
        else:
            return self.handle_general_query(query)