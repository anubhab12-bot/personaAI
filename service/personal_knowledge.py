from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class PersonalKnowledgeBase:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = {}
        self.embeddings = {}
    
    def add_personal_data(self, json_data: dict):
        """Add personal data to the knowledge base"""
        for key, value in json_data.items():
            self.knowledge_base[key] = {
                'content': str(value),
                'context': f"{key}: {value}"
            }
            self.embeddings[key] = self.model.encode(str(value))
    
    def get_relevant_context(self, query: str, top_k: int = 2):
        """Get relevant personal context for a query"""
        query_embedding = self.model.encode(query)
        similarities = {}
        
        for key, stored_embedding in self.embeddings.items():
            similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
            similarities[key] = similarity
        
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        context = ""
        for key, _ in sorted_items:
            context += f"{self.knowledge_base[key]['context']}. "
        
        return context
    
    def get_value(self, key: str, default: str = "") -> str:
        """
        Safely get a value from the knowledge base
        
        Args:
            key: The key to look up
            default: Default value if key not found
            
        Returns:
            str: The value or default if not found
        """
        if key in self.knowledge_base:
            print(key)
            return self.knowledge_base[key]['content']
        return default