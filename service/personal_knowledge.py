import json
from typing import Dict, Any, List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PersonalKnowledgeBase:
    def __init__(self, knowledge_path: Optional[str] = None):
        self.knowledge = {}
        
        # Load knowledge if path is provided
        if knowledge_path:
            self.load_knowledge(knowledge_path)
        
        try:
            # Initialize embeddings model for semantic search
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Vector store will be initialized after knowledge is loaded
            self.vector_store = None
        except Exception as e:
            print(f"Error initializing embeddings model: {e}")
            self.embeddings = None

    def load_knowledge(self, knowledge_path: str) -> None:
        """Load knowledge from a JSON file"""
        try:
            with open(knowledge_path, 'r') as f:
                self.knowledge = json.load(f)
            print(f"Loaded knowledge from {knowledge_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading knowledge from {knowledge_path}: {e}")
            self.knowledge = {}
    
    def add_personal_data(self, data: Dict[str, Any]) -> None:
        """Add personal data to the knowledge base"""
        self.knowledge.update(data)
        # Initialize or update vector store
        if self.embeddings:
            try:
                self._initialize_vector_store()
            except Exception as e:
                print(f"Error initializing vector store: {e}")
    
    def _initialize_vector_store(self) -> None:
        """Convert knowledge to documents and create a vector store"""
        if not self.knowledge:
            print("Warning: No knowledge to vectorize")
            return
            
        # Convert knowledge to documents
        documents = []
        for category, content in self.knowledge.items():
            if isinstance(content, list):
                for item in content:
                    documents.append(Document(
                        page_content=item,
                        metadata={"category": category}
                    ))
            elif isinstance(content, dict):
                for subcategory, text in content.items():
                    documents.append(Document(
                        page_content=str(text),
                        metadata={"category": category, "subcategory": subcategory}
                    ))
            else:
                documents.append(Document(
                    page_content=str(content),
                    metadata={"category": category}
                ))
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        print(f"Created vector store with {len(split_docs)} document chunks")
    
    def get_relevant_context(self, query: str, category: Optional[str] = None, top_k: int = 5) -> str:
        """Get relevant context based on the query"""
        if not self.vector_store or not self.embeddings:
            return "No knowledge available."
            
        try:
            # Prepare search kwargs
            search_kwargs = {"k": top_k}
            if category:
                search_kwargs["filter"] = {"category": category}
            
            # Get relevant documents
            docs = self.vector_store.similarity_search(query, **search_kwargs)
            
            # Format as string
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            print(f"Error getting relevant context: {e}")
            return "Error retrieving context."
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a specific value from the knowledge base"""
        keys = key.split('.')
        value = self.knowledge
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def add_knowledge(self, category: str, content: Any) -> None:
        """Add new knowledge to the knowledge base"""
        self.knowledge[category] = content
        # Update vector store
        if self.embeddings:
            try:
                self._initialize_vector_store()
            except Exception as e:
                print(f"Error updating vector store: {e}")
    
    def save_knowledge(self, path: str) -> None:
        """Save the knowledge base to a file"""
        try:
            with open(path, 'w') as f:
                json.dump(self.knowledge, f, indent=2)
            print(f"Saved knowledge to {path}")
        except Exception as e:
            print(f"Error saving knowledge to {path}: {e}")