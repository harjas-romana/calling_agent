import requests
import json
import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TravelRAGLayer:
    """
    Advanced Retrieval-Augmented Generation layer for Harjas Travels' AI calling agent.
    
    Features:
    - Semantic search with sentence embeddings
    - Conversation memory
    - Knowledge base management
    - API integration with OpenRouter
    - Comprehensive logging
    """
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "deepseek/deepseek-prover-v2:free",
        conversation_memory: int = 5,
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.conversation_memory = conversation_memory
        self.logger = self._setup_logging()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.knowledge_base = self._initialize_knowledge_base()
        self._precompute_embeddings()
        
        # Validate API connection
        self._validate_api_connection()

        print(f"API Key: {self.api_key}")  # Add this line in the __init__ method of TravelRAGLayer


    def _setup_logging(self):
        """Configure logging system."""
        logger = logging.getLogger("TravelRAGLayer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(f"logs/travel_rag_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(console_handler)
        
        return logger

    def _initialize_knowledge_base(self) -> Dict:
        """Initialize with Harjas Travels specific knowledge."""
        return {
            "agency_info": {
                "name": "Harjas Travels",
                "location": "1250 King Street West, Toronto, Ontario, Canada",
                "services": [
                    "International flight bookings",
                    "South Asian destination specialists",
                    "Student travel packages",
                    "Family reunion travel planning",
                    "Religious pilgrimage tours"
                ],
                "popular_countries": [
                    "India", "Pakistan", "United Arab Emirates", "Canada", 
                    "United States", "United Kingdom", "Australia"
                ],
                "payment_methods": [
                    "Visa", "Mastercard", "American Express", 
                    "Interac e-Transfer", "Bank wire transfer"
                ],
                "cancellation_policy": {
                    "flights": "Subject to airline policies",
                    "hotels": "Free cancellation up to 72 hours before",
                    "tours": "Full refund if cancelled 21+ days prior"
                }
            },
            "faqs": [
                {
                    "question": "What documents do I need for international travel?",
                    "answer": "You'll typically need a valid passport, visa (depending on destination), and any required health documents."
                },
                {
                    "question": "Do you offer travel insurance?",
                    "answer": "Yes, we offer comprehensive travel insurance covering medical emergencies, trip cancellation, and lost baggage."
                }
            ],
            "promotions": [
                {
                    "name": "Early Bird Special",
                    "details": "Book 6 months in advance for 15% off selected destinations"
                }
            ]
        }

    def _precompute_embeddings(self):
        """Precompute embeddings for all knowledge base content."""
        self.embeddings = {}
        
        # Embed all FAQ questions and answers
        self.embeddings['faqs'] = []
        for faq in self.knowledge_base['faqs']:
            question_embedding = self.embedding_model.encode(faq['question'])
            answer_embedding = self.embedding_model.encode(faq['answer'])
            self.embeddings['faqs'].append({
                'question': faq['question'],
                'answer': faq['answer'],
                'question_embedding': question_embedding,
                'answer_embedding': answer_embedding
            })
            
        # Embed other key information
        for section in ['agency_info', 'promotions']:
            self.embeddings[section] = []
            if isinstance(self.knowledge_base[section], list):
                for item in self.knowledge_base[section]:
                    if isinstance(item, dict):
                        text = json.dumps(item)
                    else:
                        text = str(item)
                    self.embeddings[section].append({
                        'text': text,
                        'embedding': self.embedding_model.encode(text)
                    })

    def _validate_api_connection(self):
        """Validate connection to OpenRouter API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": self.model, "messages": [{"role": "user", "content": "test"}]},
                timeout=5
            )
            if response.status_code != 200:
                raise ConnectionError(f"API connection failed: {response.text}")
            self.logger.info("API connection validated")
        except Exception as e:
            self.logger.error(f"API validation error: {str(e)}")
            raise

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Perform semantic search across knowledge base."""
        query_embedding = self.embedding_model.encode(query)
        results = []
        
        # Search FAQs
        for faq in self.embeddings['faqs']:
            question_score = cosine_similarity(
                [query_embedding],
                [faq['question_embedding']
            ])[0][0]
            answer_score = cosine_similarity(
                [query_embedding],
                [faq['answer_embedding']]
            )[0][0]
            results.append({
                'text': f"Q: {faq['question']}\nA: {faq['answer']}",
                'score': max(question_score, answer_score),
                'type': 'faq'
            })
        
        # Search other sections
        for section in ['agency_info', 'promotions']:
            for item in self.embeddings[section]:
                score = cosine_similarity(
                    [query_embedding],
                    [item['embedding']]
                )[0][0]
                results.append({
                    'text': item['text'],
                    'score': score,
                    'type': section
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Tuple[str, List[Dict]]:
        """Generate response using RAG approach."""
        if conversation_history is None:
            conversation_history = []
        
        # Retrieve relevant context
        context_results = self.semantic_search(query)
        context = "\n".join([res['text'] for res in context_results])
        
        # Prepare messages for LLM
        messages = [
            {
                "role": "system",
                "content": f"""You are an AI assistant for Harjas Travels. Use this context to answer:
                {context}
                
                Guidelines:
                - Be polite and professional
                - Only provide information you're confident about
                - Offer to connect to human agent if unsure
                - Keep responses clear and concise"""
            }
        ]
        
        # Add conversation history
        messages.extend(conversation_history[-self.conversation_memory:])
        messages.append({"role": "user", "content": query})
        
        # Call OpenRouter API
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result['choices'][0]['message']['content']
                
                # Update conversation history
                updated_history = conversation_history + [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": assistant_response}
                ]
                
                return assistant_response, updated_history[-self.conversation_memory*2:]
            
            error_msg = f"API error: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            return "I'm having trouble processing your request. Please try again later.", conversation_history
            
        except Exception as e:
            self.logger.error(f"Generation error: {str(e)}")
            return "I encountered an error processing your request.", conversation_history

    def update_knowledge_base(self, new_data: Dict, section: str = None):
        """Update knowledge base with new information."""
        try:
            if section:
                # Handle nested updates
                keys = section.split('.')
                current = self.knowledge_base
                for key in keys[:-1]:
                    current = current.setdefault(key, {})
                current[keys[-1]] = new_data
            else:
                self.knowledge_base.update(new_data)
            
            # Recompute embeddings
            self._precompute_embeddings()
            self.logger.info("Knowledge base updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Knowledge base update failed: {str(e)}")
            return False

    def interactive_demo(self):
        """Run interactive demo of the RAG system."""
        print("Harjas Travels AI Assistant - Interactive Demo")
        print("Type 'exit' to quit\n")
        
        history = []
        while True:
            query = input("Customer: ")
            if query.lower() == 'exit':
                break
                
            response, history = self.generate_response(query, history)
            print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    # Get API key from environment or prompt
    api_key = os.getenv("OPENROUTER_API_KEY") or input("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")
    
    # Initialize and run demo
    rag = TravelRAGLayer(api_key)
    rag.interactive_demo()