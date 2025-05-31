import requests
import json
import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import re

class RAGLayer:
    """
    Enhanced Retrieval-Augmented Generation layer for Romana Restaurant's AI calling agent.
    
    This class provides a complete solution for building an AI assistant that can:
    1. Store and manage restaurant-specific knowledge
    2. Retrieve relevant information based on semantic queries
    3. Generate human-like responses using AI models
    4. Maintain conversation history and context
    
    Attributes:
        api_key (str): OpenRouter API key for accessing language models
        model (str): The language model identifier to use
        base_url (str): Base URL for the OpenRouter API
        knowledge_base (Dict): Dictionary containing all restaurant information
        logger (logging.Logger): Logger for tracking operations
        conversation_memory (int): Number of conversation turns to remember
    """
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "deepseek/deepseek-prover-v2:free",
        conversation_memory: int = 5
    ):
        """
        Initialize the RAG layer with API credentials and knowledge base.
        
        Args:
            api_key (str): OpenRouter API key
            model (str): Model identifier for OpenRouter
            conversation_memory (int): Number of conversation turns to remember
        
        Raises:
            ValueError: If API key is empty or invalid
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Valid API key is required")
        
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.knowledge_base = self._initialize_knowledge_base()
        self.logger = self._setup_logging()
        self.conversation_memory = conversation_memory
        
        # Validate API key with a simple test call
        self._validate_api_connection()
        
    def _validate_api_connection(self) -> None:
        """
        Validate the API connection by making a small test request.
        
        Raises:
            ConnectionError: If unable to connect to the API
        """
        try:
            # Simple test message to verify connection
            test_message = [{"role": "user", "content": "Hello"}]
            
            # Make a minimal request to verify API key works
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": test_message,
                    "max_tokens": 10
                },
                timeout=5  # Set a reasonable timeout
            )
            
            if response.status_code != 200:
                self.logger.error(f"API validation failed: {response.status_code} - {response.text}")
                raise ConnectionError(f"Failed to connect to OpenRouter API: {response.status_code}")
                
            self.logger.info("API connection validated successfully")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API connection error: {str(e)}")
            raise ConnectionError(f"Failed to connect to OpenRouter API: {str(e)}")
        
        
    
    def _setup_logging(self) -> logging.Logger:
        """
        Configure logging system for the RAG layer.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Configure logger with file and console handlers
        logger = logging.getLogger("RAGLayer")
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            # File handler - logs everything to a date-stamped file
            file_handler = logging.FileHandler(
                f"logs/rag_layer_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(logging.INFO)
            
            # Console handler - only shows warnings and errors
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            
            # Create formatter and add to handlers
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """
        Initialize the restaurant-specific knowledge base with structured data.
        
        Returns:
            Dict: Complete knowledge base structure
        """
        return {
            "restaurant_info": {
                "name": "Romana Restaurant",
                "cuisine": "Italian",
                "description": "Authentic Italian cuisine serving homemade pasta and wood-fired pizzas",
                "location": "Toronto, Canada",
                "phone": "(555) 123-4567",
                "email": "reservations@romanarestaurant.com",
                "website": "www.romanarestaurant.com",
                "hours": {
                    "monday": "11:00 AM - 10:00 PM",
                    "tuesday": "11:00 AM - 10:00 PM",
                    "wednesday": "11:00 AM - 10:00 PM",
                    "thursday": "11:00 AM - 10:00 PM",
                    "friday": "11:00 AM - 11:00 PM",
                    "saturday": "10:00 AM - 11:00 PM",
                    "sunday": "10:00 AM - 10:00 PM"
                },
                "popular_dishes": [
                    {
                        "name": "Spaghetti Carbonara",
                        "description": "Classic carbonara with pancetta, egg, black pepper, and Pecorino Romano",
                        "price": 16.99,
                        "allergens": ["gluten", "dairy", "eggs"]
                    },
                    {
                        "name": "Margherita Pizza",
                        "description": "Traditional pizza with San Marzano tomatoes, fresh mozzarella, and basil",
                        "price": 14.99,
                        "allergens": ["gluten", "dairy"]
                    },
                    {
                        "name": "Lasagna Bolognese",
                        "description": "Layered pasta with beef ragù, béchamel sauce, and Parmigiano",
                        "price": 18.99,
                        "allergens": ["gluten", "dairy", "eggs"]
                    },
                    {
                        "name": "Tiramisu",
                        "description": "Classic Italian dessert with espresso-soaked ladyfingers and mascarpone",
                        "price": 9.99,
                        "allergens": ["gluten", "dairy", "eggs"]
                    },
                    {
                        "name": "Risotto al Funghi",
                        "description": "Creamy arborio rice with wild mushrooms and Parmigiano",
                        "price": 17.99,
                        "allergens": ["dairy"]
                    }
                ],
                "specials": {
                    "monday": "20% off all pasta dishes",
                    "tuesday": "Wine pairing special - half-price wine by the glass",
                    "wednesday": "Family meal deal - 4 people eat for $60",
                    "thursday": "Date night package - 3-course meal for two $65",
                    "friday": "Happy hour 4-6 PM - $5 appetizers and drinks"
                },
                "policies": {
                    "reservations": "Reservations recommended, especially on weekends",
                    "cancellations": "24-hour cancellation policy",
                    "dress_code": "Business casual",
                    "parking": "Valet available for $10, street parking also available",
                    "pets": "Service animals only",
                    "payment": "We accept all major credit cards, no checks"
                },
                "dietary_accommodations": {
                    "gluten_free": True,
                    "vegetarian": True,
                    "vegan": True,
                    "dairy_free": True,
                    "nut_free": True
                }
            },
            "menu_categories": [
                "Appetizers", "Pasta", "Pizza", "Main Courses", "Desserts", "Beverages"
            ],
            "faqs": [
                {
                    "question": "Do you offer gluten-free options?",
                    "answer": "Yes, we have gluten-free pasta and pizza crust available for an additional $2."
                },
                {
                    "question": "Is there a kids' menu?",
                    "answer": "Yes, we offer a children's menu with smaller portions priced at $9.99, including a drink and dessert."
                },
                {
                    "question": "Can you accommodate food allergies?",
                    "answer": "Yes, please inform your server of any allergies when ordering. Our kitchen can accommodate most dietary restrictions."
                },
                {
                    "question": "Do you have outdoor seating?",
                    "answer": "Yes, we have a beautiful patio that's open seasonally, weather permitting."
                },
                {
                    "question": "Do you take reservations?",
                    "answer": "Yes, we recommend reservations, especially for weekend dining. You can book online or call us at (555) 123-4567."
                },
                {
                    "question": "Is there a corkage fee?",
                    "answer": "Yes, we allow outside wine with a $25 corkage fee per bottle."
                }
            ],
            "last_updated": datetime.now().isoformat()
        }
    
    def update_knowledge_base(
        self, 
        new_data: Dict[str, Any], 
        section: Optional[str] = None,
        merge: bool = True
    ) -> bool:
        """
        Update the knowledge base with new information.
        
        Args:
            new_data (Dict): New data to add/update
            section (str, optional): Specific section to update. If None, updates root.
            merge (bool): If True, merges data with existing; if False, replaces it
            
        Returns:
            bool: True if update was successful, False otherwise
            
        Example:
            >>> rag.update_knowledge_base({"opens_at": "10:30 AM"}, "restaurant_info/hours/monday")
        """
        try:
            # Handle nested sections with path notation (e.g., "restaurant_info/hours/monday")
            if section:
                path_parts = section.split('/')
                
                # Navigate to the correct section
                current = self.knowledge_base
                for i, part in enumerate(path_parts[:-1]):
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                last_part = path_parts[-1]
                
                # Update or replace the data
                if merge and last_part in current and isinstance(current[last_part], dict) and isinstance(new_data, dict):
                    current[last_part].update(new_data)
                else:
                    current[last_part] = new_data
            else:
                # Update root level
                if merge:
                    self.knowledge_base.update(new_data)
                else:
                    for key in new_data:
                        self.knowledge_base[key] = new_data[key]
            
            # Update the last_updated timestamp
            self.knowledge_base["last_updated"] = datetime.now().isoformat()
            
            self.logger.info(f"Knowledge base updated successfully at section: {section if section else 'root'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating knowledge base: {str(e)}")
            return False
    
    def get_knowledge_base(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve information from the knowledge base.
        
        Args:
            section (str, optional): Specific section to retrieve. If None, returns complete knowledge base.
            
        Returns:
            Dict: The requested knowledge base section or complete knowledge base
            
        Example:
            >>> hours = rag.get_knowledge_base("restaurant_info/hours")
        """
        try:
            if not section:
                return self.knowledge_base
            
            # Handle nested sections with path notation
            path_parts = section.split('/')
            current = self.knowledge_base
            
            for part in path_parts:
                if part not in current:
                    self.logger.warning(f"Section {section} not found in knowledge base")
                    return {}
                current = current[part]
                
            return current
            
        except Exception as e:
            self.logger.error(f"Error retrieving from knowledge base: {str(e)}")
            return {}
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple text similarity score between two strings.
        
        Args:
            text1 (str): First text string
            text2 (str): Second text string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        # Avoid division by zero
        return intersection / max(union, 1)
    
    def retrieve_relevant_context(self, query: str, threshold: float = 0.1) -> str:
        """
        Retrieve relevant context from knowledge base based on user query.
        Uses simple text similarity to find matching information.
        
        Args:
            query (str): User's input query
            threshold (float): Minimum similarity score to include context (0-1)
            
        Returns:
            str: Retrieved context as a formatted string
        """
        try:
            # Clean and normalize the query
            clean_query = query.lower().strip()
            context_parts = []
            
            # Score different parts of the knowledge base
            # Extract and score restaurant info
            for key, value in self.knowledge_base['restaurant_info'].items():
                if isinstance(value, str):
                    similarity = self._simple_text_similarity(clean_query, value)
                    if similarity > threshold or key.lower() in clean_query:
                        context_parts.append({
                            "text": f"{key.replace('_', ' ').title()}: {value}",
                            "score": similarity + (0.2 if key.lower() in clean_query else 0)
                        })
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    for sub_key, sub_value in value.items():
                        combined_text = f"{sub_key}: {sub_value}"
                        similarity = self._simple_text_similarity(clean_query, combined_text)
                        if similarity > threshold or sub_key.lower() in clean_query:
                            context_parts.append({
                                "text": f"{key.replace('_', ' ').title()} - {sub_key.replace('_', ' ').title()}: {sub_value}",
                                "score": similarity + (0.2 if sub_key.lower() in clean_query else 0)
                            })
                elif isinstance(value, list):
                    # Handle lists of items
                    for item in value:
                        if isinstance(item, dict):
                            # Handle list of dictionaries (like menu items)
                            item_text = json.dumps(item)
                            similarity = self._simple_text_similarity(clean_query, item_text)
                            if similarity > threshold:
                                context_parts.append({
                                    "text": f"{key.replace('_', ' ').title()}: {item}",
                                    "score": similarity
                                })
                        else:
                            # Handle simple list items
                            similarity = self._simple_text_similarity(clean_query, str(item))
                            if similarity > threshold:
                                context_parts.append({
                                    "text": f"{key.replace('_', ' ').title()}: {item}",
                                    "score": similarity
                                })
            
            # Process FAQs - these are especially important for customer questions
            for faq in self.knowledge_base['faqs']:
                question_similarity = self._simple_text_similarity(clean_query, faq['question'])
                answer_similarity = self._simple_text_similarity(clean_query, faq['answer'])
                max_similarity = max(question_similarity, answer_similarity)
                
                if max_similarity > threshold:
                    context_parts.append({
                        "text": f"FAQ: {faq['question']} - {faq['answer']}",
                        "score": max_similarity + 0.1  # Boost FAQs slightly
                    })
            
            # Sort by relevance score
            context_parts.sort(key=lambda x: x["score"], reverse=True)
            
            # Take the top 5 most relevant pieces of context
            top_contexts = [item["text"] for item in context_parts[:5]]
            
            return "\n".join(top_contexts) if top_contexts else "No specific context found for your query."
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            return "Error retrieving context information."
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract potentially important keywords from text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[str]: List of extracted keywords
        """
        # Remove punctuation and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = cleaned_text.split()
        
        # Remove common stopwords
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
            'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
            'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'should', 'now'
        }
        
        # Extract words that aren't stopwords and are at least 3 characters
        keywords = [word for word in words if word not in stopwords and len(word) >= 3]
        
        # Return unique keywords
        return list(set(keywords))
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance the user query with additional search terms.
        
        Args:
            query (str): User's original query
            
        Returns:
            str: Enhanced query with additional relevant terms
        """
        # Extract keywords from the query
        keywords = self._extract_keywords(query)
        
        # Map common synonyms and related terms for restaurant context
        restaurant_term_mappings = {
            'hour': ['hours', 'open', 'closing', 'schedule', 'time'],
            'reservation': ['book', 'booking', 'reserve', 'table'],
            'menu': ['food', 'dish', 'dishes', 'eat', 'cuisine'],
            'price': ['cost', 'expensive', 'cheap', 'affordable'],
            'allerg': ['allergic', 'allergen', 'allergy'],
            'vegetarian': ['vegan', 'plant', 'meat'],
            'park': ['parking', 'valet', 'car'],
            'kid': ['child', 'children', 'family', 'baby'],
            'gluten': ['celiac', 'wheat', 'pasta'],
            'special': ['deal', 'discount', 'offer', 'promotion'],
            'dessert': ['sweet', 'cake', 'ice cream', 'tiramisu'],
            'wine': ['alcohol', 'drink', 'beverage'],
            'pizza': ['pie', 'margherita', 'pepperoni']
        }
        
        # Add related terms to the query
        enhanced_terms = set(keywords)
        for keyword in keywords:
            for term, related in restaurant_term_mappings.items():
                if term in keyword:
                    enhanced_terms.update(related)
        
        # Combine original query with enhanced terms
        return query + " " + " ".join(enhanced_terms)
    
    def generate_response(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 350
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Generate a response using the RAG model.
        
        Args:
            query (str): User's input query
            conversation_history (List[Dict], optional): Previous conversation context
            temperature (float): Control randomness (0-1)
            max_tokens (int): Maximum response length
            
        Returns:
            Tuple[str, List[Dict]]: Generated response and updated conversation history
        """
        try:
            # Initialize conversation history if None
            if conversation_history is None:
                conversation_history = []
            
            # Enhance the query for better context retrieval
            enhanced_query = self.enhance_query(query)
            
            # Retrieve relevant context
            context = self.retrieve_relevant_context(enhanced_query)
            
            # Prepare messages for the model
            messages = []
            
            # System message with instructions and context
            system_message = {
                "role": "system",
                "content": f"""You are an AI assistant for Romana Restaurant, an authentic Italian eatery. 
                Use the following context to answer the user's question. Be polite, professional, and helpful.
                
                Context:
                {context}
                
                Only provide information that is contained in the context or can be directly inferred from it.
                If you don't know the answer, politely say you don't have that information and offer to connect them to a human.
                Keep your answers concise but complete.
                """
            }
            messages.append(system_message)
            
            # Add limited conversation history
            if conversation_history:
                messages.extend(conversation_history[-self.conversation_memory:])
            
            # Add current user query
            user_message = {"role": "user", "content": query}
            messages.append(user_message)
            
            # Call OpenRouter API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result['choices'][0]['message']['content']
                
                # Update conversation history
                conversation_history.append(user_message)
                conversation_history.append({"role": "assistant", "content": assistant_response})
                
                # Maintain conversation history size
                if len(conversation_history) > self.conversation_memory * 2:
                    conversation_history = conversation_history[-self.conversation_memory * 2:]
                
                return assistant_response, conversation_history
            else:
                error_msg = f"API call failed: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                fallback_response = "I'm having trouble processing your request. Please try again later."
                
                # Still add to conversation history
                conversation_history.append(user_message)
                conversation_history.append({"role": "assistant", "content": fallback_response})
                
                return fallback_response, conversation_history
                
        except requests.exceptions.Timeout:
            timeout_msg = "Request timed out. The server might be busy."
            self.logger.error(timeout_msg)
            return timeout_msg, conversation_history
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            return "I encountered an error. Please try again.", conversation_history
    
    def backup_knowledge_base(self, filename: str = "knowledge_backup.json") -> bool:
        """
        Save the current knowledge base to a backup file.
        
        Args:
            filename (str): Name of the backup file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create backup directory if it doesn't exist
            backup_dir = "backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{backup_dir}/{timestamp}_{filename}"
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=4)
                
            self.logger.info(f"Knowledge base backed up to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error backing up knowledge base: {str(e)}")
            return False
    
    def restore_knowledge_base(self, filename: str) -> bool:
        """
        Restore knowledge base from a backup file.
        
        Args:
            filename (str): Path to the backup file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(filename):
                self.logger.error(f"Backup file not found: {filename}")
                return False
                
            # Read from file
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Validate data format
            if not isinstance(data, dict) or 'restaurant_info' not in data:
                self.logger.error("Invalid backup file format")
                return False
                
            # Update knowledge base
            self.knowledge_base = data
            self.logger.info(f"Knowledge base restored from {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring knowledge base: {str(e)}")
            return False


def interactive_demo():
    """
    Run an interactive demo of the RAG Layer.
    """
    import os
    
    # Get API key from environment or prompt
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenRouter API key: ")
        
    if not api_key:
        print("API key is required to run the demo.")
        return
        
    # Initialize RAG layer
    print("Initializing RAG Layer...")
    
    try:
        rag = RAGLayer(api_key)
        print("RAG Layer initialized successfully!")
        
        # Start interactive session
        print("\n===== Romana Restaurant AI Assistant Demo =====")
        print("Type 'exit' to quit, 'help' for commands")
        
        conversation_history = []
        
        while True:
            query = input("\nUser: ")
            
            if query.lower() == 'exit':
                print("Thank you for using Romana Restaurant AI Assistant!")
                break
                
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("- 'exit': Quit the demo")
                print("- 'help': Show this help message")
                print("- 'backup': Backup the knowledge base")
                print("- 'clear': Clear conversation history")
                print("- Any other input: Ask a question about Romana Restaurant")
                
            elif query.lower() == 'backup':
                if rag.backup_knowledge_base():
                    print("Knowledge base backed up successfully!")
                else:
                    print("Failed to backup knowledge base.")
                    
            elif query.lower() == 'clear':
                conversation_history = []
                print("Conversation history cleared.")
                
            else:
                print("Generating response...")
                response, conversation_history = rag.generate_response(query, conversation_history)
                print(f"\nAssistant: {response}")
                
    except Exception as e:
        print(f"Error running demo: {str(e)}")


if __name__ == "__main__":
    # Run interactive demo
    interactive_demo()