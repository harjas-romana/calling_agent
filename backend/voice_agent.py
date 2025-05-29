import requests
import json
import os
import logging
import sounddevice as sd
import soundfile as sf
import io
import numpy as np
import re
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from rag_layer import RAGLayer
import speech_recognition as sr

class VoiceAgent:
    """
    Enhanced Voice Agent for Romana Restaurant with complete reservation,
    ordering, and customer service capabilities.
    """
    
    def __init__(
        self, 
        elevenlabs_api_key: str,
        rag_layer: RAGLayer,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel's voice
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        audio_device: Optional[int] = None
    ):
        self.elevenlabs_api_key = elevenlabs_api_key
        self.rag_layer = rag_layer
        self.base_url = "https://api.elevenlabs.io/v1"
        self.logger = self._setup_logging()
        self.audio_device = audio_device
        
        # Voice configuration
        self.voice_settings = {
            "voice_id": voice_id,
            "stability": stability,
            "similarity_boost": similarity_boost,
            "model_id": "eleven_monolingual_v1"
        }
        
        # Restaurant operational data
        self.reservations = []
        self.orders = []
        self.current_order = None
        self.current_reservation = None
        
        # Restaurant configuration
        self.operating_hours = {
            "Monday": "11:00 AM - 10:00 PM",
            "Tuesday": "11:00 AM - 10:00 PM",
            "Wednesday": "11:00 AM - 10:00 PM",
            "Thursday": "11:00 AM - 10:00 PM",
            "Friday": "11:00 AM - 11:00 PM",
            "Saturday": "10:00 AM - 11:00 PM",
            "Sunday": "10:00 AM - 10:00 PM"
        }
        
        self._validate_api_connection()
        self._list_audio_devices()

    def _validate_api_connection(self):
        """Validate the connection to ElevenLabs API."""
        try:
            headers = {"xi-api-key": self.elevenlabs_api_key}
            response = requests.get(f"{self.base_url}/voices", headers=headers)
            if response.status_code != 200:
                raise RuntimeError(f"API connection failed: {response.status_code}")
            self.logger.info("API connection validated successfully")
        except Exception as error:
            self.logger.error(f"API validation failed: {str(error)}")
            raise RuntimeError("Could not validate API connection")

    def _list_audio_devices(self):
        """List available audio devices and verify audio configuration."""
        try:
            sd.query_devices()
            self.logger.info("Audio devices initialized successfully")
        except Exception as error:
            self.logger.error(f"Audio device initialization failed: {str(error)}")
            raise RuntimeError("Could not initialize audio devices")

    def _setup_logging(self):
        """Configure logging system."""
        os.makedirs("logs", exist_ok=True)
        logger = logging.getLogger("VoiceAgent")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            file_handler = logging.FileHandler(f"logs/voice_agent_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler.setFormatter(formatter)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    def text_to_speech(self, text: str, output_file: Optional[str] = None) -> Optional[bytes]:
        """Convert text to speech using ElevenLabs API."""
        try:
            headers = {
                "xi-api-key": self.elevenlabs_api_key,
                "Content-Type": "application/json",
                "accept": "audio/mpeg"
            }
            
            data = {
                "text": text,
                "voice_settings": {
                    "stability": self.voice_settings["stability"],
                    "similarity_boost": self.voice_settings["similarity_boost"]
                },
                "model_id": self.voice_settings["model_id"]
            }
            
            response = requests.post(
                f"{self.base_url}/text-to-speech/{self.voice_settings['voice_id']}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                audio_data = response.content
                if output_file:
                    with open(output_file, 'wb') as f:
                        f.write(audio_data)
                    return None
                return audio_data
            else:
                raise RuntimeError(f"TTS failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error in TTS conversion: {str(e)}")
            raise RuntimeError(f"Error in TTS conversion: {str(e)}")

    def play_audio(self, audio_data: bytes) -> None:
        """Play audio from bytes."""
        try:
            audio_file = io.BytesIO(audio_data)
            data, samplerate = sf.read(audio_file)
            if len(data.shape) == 1:
                data = np.column_stack((data, data))
            sd.play(data, samplerate, device=self.audio_device)
            sd.wait()
        except Exception as e:
            self.logger.error(f"Error playing audio: {str(e)}")
            raise RuntimeError(f"Error playing audio: {str(e)}")

    # Enhanced Conversation Handling
    def handle_conversation(self, query: str, conversation_history: List[Dict]) -> tuple:
        """
        Main conversation handler that routes to specific functions based on context.
        Returns tuple: (response_text, audio_data, updated_history)
        """
        try:
            # Check if we're in the middle of a reservation
            if self.current_reservation and not self.current_reservation.get('completed', False):
                return self._handle_reservation_flow(query, conversation_history)
            
            # Check if we're in the middle of an order
            if self.current_order and not self.current_order.get('completed', False):
                return self._handle_ordering_flow(query, conversation_history)
            
            # Check for specific intents
            query_lower = query.lower()
            
            if any(word in query_lower for word in ["reservation", "book", "table", "reserve"]):
                return self._start_reservation(query, conversation_history)
            
            elif any(word in query_lower for word in ["order", "menu", "food", "dish", "eat", "hungry"]):
                return self._start_ordering(query, conversation_history)
            
            elif any(word in query_lower for word in ["hours", "open", "close", "timing", "schedule"]):
                response = self._get_operating_hours()
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
            
            elif any(word in query_lower for word in ["feedback", "review", "experience", "comment"]):
                return self._handle_feedback(query, conversation_history)
            
            elif any(word in query_lower for word in ["help", "commands", "options", "what can you do"]):
                response = self._get_available_commands()
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
            
            elif any(word in query_lower for word in ["specials", "today", "chef", "recommend", "popular"]):
                return self._get_daily_specials(conversation_history)
            
            elif any(word in query_lower for word in ["location", "address", "directions", "find"]):
                return self._get_location_info(conversation_history)
            
            else:
                # Default to RAG response
                return self.rag_layer.generate_response(query, conversation_history)
                
        except Exception as e:
            self.logger.error(f"Error in conversation handling: {str(e)}")
            error_msg = "I'm sorry, I encountered an error processing your request. Could you please try again?"
            try:
                audio = self.text_to_speech(error_msg)
                return error_msg, audio, conversation_history
            except:
                return error_msg, None, conversation_history

    # Reservation System
    def _start_reservation(self, query: str, conversation_history: List[Dict]) -> tuple:
        """Initiate reservation process."""
        self.current_reservation = {
            "completed": False,
            "step": "party_size",
            "data": {}
        }
        
        response = "Thank you for choosing Romana Restaurant! Please tell me how many people will be dining with us? Just say a number."
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    def _handle_reservation_flow(self, query: str, conversation_history: List[Dict]) -> tuple:
        """Handle multi-step reservation process."""
        current_step = self.current_reservation["step"]
        
        if current_step == "party_size":
            # Extract number from text
            party_size = self._extract_number_from_text(query)
            
            if party_size is not None and party_size > 0:
                self.current_reservation["data"]["party_size"] = party_size
                self.current_reservation["step"] = "date"
                
                response = f"Great! We'll reserve for {party_size} people. What date would you like to dine with us? You can say tomorrow, Friday, or a specific date."
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
            else:
                response = "I need to know how many people will be dining. Please say just a number, like 'four' or 'six'."
                audio = self.text_to_speech(response)
                return response, audio, conversation_history
        
        elif current_step == "date":
            try:
                parsed_date = self._parse_date(query)
                if parsed_date < datetime.now().date():
                    response = "I'm sorry, we can't make reservations for dates in the past. Please choose a future date."
                    audio = self.text_to_speech(response)
                    return response, audio, conversation_history
                
                self.current_reservation["data"]["date"] = parsed_date.strftime("%Y-%m-%d")
                self.current_reservation["step"] = "time"
                
                response = "What time would you like to reserve? Our hours are 11AM to 10PM."
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
                
            except Exception as e:
                response = "I couldn't understand that date. Please say something like 'tomorrow', 'this Friday', or 'May 20th'."
                audio = self.text_to_speech(response)
                return response, audio, conversation_history
        
        elif current_step == "time":
            try:
                parsed_time = self._parse_time(query)
                self.current_reservation["data"]["time"] = parsed_time.strftime("%I:%M %p")
                self.current_reservation["step"] = "name"
                
                response = "Perfect! What name should I put the reservation under?"
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
                
            except Exception as e:
                response = "Please tell me a valid time between 11AM and 10PM, like 'seven thirty PM' or '12:45 PM'."
                audio = self.text_to_speech(response)
                return response, audio, conversation_history
        
        elif current_step == "name":
            self.current_reservation["data"]["name"] = query
            self.current_reservation["step"] = "phone"
            
            response = "Thank you. Could I also have a contact phone number in case we need to reach you?"
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
            
        elif current_step == "phone":
            # Simple validation - we're just checking if there are digits
            if any(char.isdigit() for char in query):
                self.current_reservation["data"]["phone"] = query
                self.current_reservation["step"] = "confirm"
                
                # Format confirmation message
                res_data = self.current_reservation["data"]
                response = (
                    f"Let me confirm your reservation:\n"
                    f"Name: {res_data['name']}\n"
                    f"Phone: {res_data['phone']}\n"
                    f"Party Size: {res_data['party_size']}\n"
                    f"Date: {res_data['date']}\n"
                    f"Time: {res_data['time']}\n\n"
                    f"Is this information correct? Please say yes or no."
                )
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
            else:
                response = "I need a phone number with digits. Please provide a valid phone number."
                audio = self.text_to_speech(response)
                return response, audio, conversation_history
        
        elif current_step == "confirm":
            if "yes" in query.lower() or "correct" in query.lower() or "right" in query.lower():
                # Complete reservation
                self.reservations.append(self.current_reservation["data"])
                self.current_reservation["completed"] = True
                
                response = (
                    "Your reservation is confirmed! We look forward to serving you at Romana Restaurant. "
                    "Do you have any special requests or dietary restrictions we should know about?"
                )
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
            else:
                self.current_reservation["step"] = "party_size"
                response = "Let's start over. How many people will be dining with us?"
                audio = self.text_to_speech(response)
                return response, audio, conversation_history

    # Ordering System
    def _start_ordering(self, query: str, conversation_history: List[Dict]) -> tuple:
        """Initiate food ordering process."""
        self.current_order = {
            "completed": False,
            "step": "table_number",
            "items": [],
            "special_requests": ""
        }
        
        # Get menu from knowledge base
        menu = self.rag_layer.get_knowledge_base("restaurant_info/popular_dishes")
        menu_text = "\n".join([f"{item['name']} - ${item['price']}" for item in menu])
        
        response = (
            "I'd be happy to take your order. First, could you tell me your table number?\n\n"
            f"Our popular dishes today:\n{menu_text}"
        )
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    def _handle_ordering_flow(self, query: str, conversation_history: List[Dict]) -> tuple:
        """Handle multi-step ordering process."""
        current_step = self.current_order["step"]
        
        if current_step == "table_number":
            # Extract number from text
            table_num = self._extract_number_from_text(query)
            
            if table_num is not None and table_num > 0:
                self.current_order["table_number"] = table_num
                self.current_order["step"] = "item_selection"
                
                response = "Great! What would you like to order? You can say multiple items at once."
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
            else:
                response = "I need your table number. Please say just the number, like 'table five' or 'number seven'."
                audio = self.text_to_speech(response)
                return response, audio, conversation_history
        
        elif current_step == "item_selection":
            # Match items to menu
            menu = self.rag_layer.get_knowledge_base("restaurant_info/popular_dishes")
            ordered_items = []
            
            for item in menu:
                if item['name'].lower() in query.lower():
                    ordered_items.append(item)
            
            if ordered_items:
                self.current_order["items"].extend(ordered_items)
                total = sum(item['price'] for item in self.current_order["items"])
                
                item_names = ", ".join(item['name'] for item in ordered_items)
                response = (
                    f"I've added {item_names} to your order. Current total: ${total:.2f}\n"
                    "Would you like to add anything else? Please say yes or no."
                )
                
                if "no" in query.lower() or "that's it" in query.lower() or "that's all" in query.lower():
                    self.current_order["step"] = "special_requests"
                    response = "Any special requests or dietary restrictions we should know about?"
            else:
                response = "I didn't recognize those menu items. Could you please try again or ask to hear our menu?"
            
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "special_requests":
            self.current_order["special_requests"] = query
            self.current_order["step"] = "confirm_order"
            
            # Format order summary
            items_text = "\n".join(
                f"- {item['name']} (${item['price']})" 
                for item in self.current_order["items"]
            )
            total = sum(item['price'] for item in self.current_order["items"])
            
            response = (
                "Let me confirm your order:\n"
                f"Table: {self.current_order['table_number']}\n"
                f"Items:\n{items_text}\n"
                f"Special Requests: {self.current_order['special_requests']}\n"
                f"Total: ${total:.2f}\n\n"
                "Should I place this order? Please say yes or no."
            )
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "confirm_order":
            if "yes" in query.lower() or "confirm" in query.lower() or "place" in query.lower():
                # Complete order
                self.orders.append(self.current_order)
                self.current_order["completed"] = True
                
                total = sum(item['price'] for item in self.current_order["items"])
                response = (
                    f"Your order has been placed! Total amount: ${total:.2f}\n"
                    "Your food will be prepared shortly. Is there anything else I can help with today?"
                )
            else:
                self.current_order["step"] = "item_selection"
                response = "What would you like to change about your order?"
            
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history

    # Feedback System
    def _handle_feedback(self, query: str, conversation_history: List[Dict]) -> tuple:
        """Handle customer feedback collection."""
        response = (
            "Thank you for your feedback! We truly value your opinion. "
            "Could you share what you enjoyed most about your dining experience today, and if there's anything we could improve?"
        )
        
        # In a real implementation, you would store this feedback
        self.logger.info(f"Customer feedback received: {query}")
        
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    # New Feature: Daily Specials
    def _get_daily_specials(self, conversation_history: List[Dict]) -> tuple:
        """Provide information about daily specials."""
        today = datetime.now().strftime("%A")
        specials = {
            "Monday": "Mushroom Risotto with truffle oil and Tiramisu for dessert",
            "Tuesday": "Homemade Lasagna with garlic bread and Panna Cotta",
            "Wednesday": "Seafood Linguine with white wine sauce and Lemon Sorbet",
            "Thursday": "Osso Buco with saffron risotto and Cannoli",
            "Friday": "Grilled Sea Bass with Mediterranean vegetables and Chocolate Fondant",
            "Saturday": "Prime Rib with truffle mashed potatoes and Crème Brûlée",
            "Sunday": "Sunday Roast with all the trimmings and Gelato selection"
        }
        
        response = (
            f"Today's specials for {today} are: {specials[today]}. "
            "Our chef personally recommends pairing it with our house wine selection. "
            "Would you like to include any of these items in your order?"
        )
        
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    # New Feature: Location Information
    def _get_location_info(self, conversation_history: List[Dict]) -> tuple:
        """Provide restaurant location and directions."""
        response = (
            "Romana Restaurant is located at 123 Culinary Avenue, Downtown. "
            "We're right across from Central Park and just two blocks from the Main Street subway station. "
            "Free parking is available in our private lot behind the restaurant. "
            "Would you like me to send directions to your phone?"
        )
        
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    # New Feature: Available Commands
    def _get_available_commands(self) -> str:
        """Return list of available commands and features."""
        return (
            "Here are some things you can ask me:\n"
            "- 'Book a table' or 'Make a reservation'\n"
            "- 'I'd like to order food'\n"
            "- 'What are today's specials?'\n"
            "- 'What are your hours?'\n"
            "- 'Where are you located?'\n"
            "- 'I have feedback about my experience'\n"
            "- Ask any questions about our menu, ingredients, or services!"
        )

    # Utility Methods
    def _extract_number_from_text(self, text: str) -> Optional[int]:
        """Extract numeric values from text, handling both digits and word forms."""
        # First, check for digit numbers
        digit_match = re.search(r'\b(\d+)\b', text)
        if digit_match:
            return int(digit_match.group(1))
        
        # Word to number mapping
        word_to_number = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
        }
        
        # Check for word numbers
        text_lower = text.lower()
        for word, number in word_to_number.items():
            if word in text_lower:
                return number
                
        return None

    def _parse_date(self, date_str: str) -> datetime.date:
        """Parse natural language dates into datetime objects."""
        date_str = date_str.lower()
        today = datetime.now().date()
        
        # Check for common phrases
        if "today" in date_str:
            return today
        elif "tomorrow" in date_str:
            return today + timedelta(days=1)
        elif "day after tomorrow" in date_str:
            return today + timedelta(days=2)
        
        # Check for day names (e.g., "this Friday", "next Monday")
        days = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
                "friday": 4, "saturday": 5, "sunday": 6}
        
        for day_name, day_num in days.items():
            if day_name in date_str:
                today_weekday = today.weekday()
                days_until = (day_num - today_weekday) % 7
                
                # If "next" is mentioned, add a week
                if "next" in date_str:
                    days_until += 7
                
                # If days_until is 0 and not explicitly "today", assume next week
                if days_until == 0 and "today" not in date_str:
                    days_until = 7
                    
                return today + timedelta(days=days_until)
        
        # Try to parse specific date formats
        try:
            # Try YYYY-MM-DD format
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            pass
            
        try:
            # Try Month Day format (e.g., "May 20")
            months = ["january", "february", "march", "april", "may", "june", 
                      "july", "august", "september", "october", "november", "december"]
            
            for i, month in enumerate(months, 1):
                if month in date_str:
                    # Extract the day
                    day_match = re.search(r'\b(\d{1,2})(st|nd|rd|th)?\b', date_str)
                    if day_match:
                        day = int(day_match.group(1))
                        year = today.year
                        
                        # If the date is in the past, assume next year
                        date_obj = datetime(year, i, day).date()
                        if date_obj < today:
                            date_obj = datetime(year + 1, i, day).date()
                            
                        return date_obj
        except Exception:
            pass
        
        raise ValueError("Could not parse date")

    def _parse_time(self, time_str: str) -> datetime.time:
        """Parse natural language times into time objects."""
        time_str = time_str.lower()
        
        # Check for common phrases
        if "noon" in time_str:
            return datetime.strptime("12:00 PM", "%I:%M %p").time()
        elif "midnight" in time_str:
            return datetime.strptime("12:00 AM", "%I:%M %p").time()
        
        # Try to extract hour and minute information
        hour_pattern = r'(\d{1,2})'
        minute_pattern = r':(\d{1,2})'
        period_pattern = r'([ap]\.?m\.?)'
        
        # Extract hour
        hour_match = re.search(hour_pattern, time_str)
        if not hour_match:
            # Check for spoken time
            time_words = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'eleven': 11, 'twelve': 12
            }
            
            for word, number in time_words.items():
                if word in time_str:
                    hour = number
                    break
            else:
                raise ValueError("Could not identify hour")
        else:
            hour = int(hour_match.group(1))
        
        # Extract minute
        minute_match = re.search(minute_pattern, time_str)
        minute = 0
        if minute_match:
            minute = int(minute_match.group(1))
        elif "half" in time_str or "thirty" in time_str:
            minute = 30
        elif "quarter" in time_str:
            if "to" in time_str:
                minute = 45
                hour = (hour - 1) % 12
            else:  # "quarter past"
                minute = 15
        
        # Extract AM/PM
        period_match = re.search(period_pattern, time_str)
        is_pm = False
        
        if period_match:
            if period_match.group(1).startswith('p'):
                is_pm = True
        else:
            # If no explicit AM/PM, try to infer
            if "evening" in time_str or "night" in time_str or hour < 7:
                is_pm = True
            elif "afternoon" in time_str and hour < 12:
                is_pm = True
        
        # Convert to 24-hour format if PM
        if is_pm and hour < 12:
            hour += 12
        elif not is_pm and hour == 12:
            hour = 0
        
        try:
            return datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()
        except ValueError:
            raise ValueError("Could not parse time")

    def _get_operating_hours(self) -> str:
        """Return formatted operating hours."""
        hours_text = "\n".join(f"{day}: {hours}" for day, hours in self.operating_hours.items())
        return f"Our operating hours are:\n{hours_text}\n\nIs there a particular day you're planning to visit?"


def interactive_demo():
    """Run an interactive demo of the enhanced Voice Agent with verbal input."""
    import os
    import time
    
    # Get API keys
    openrouter_key = "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"
    elevenlabs_key = "sk_a643471cf3d2de658ac47648b33d8314bfe39dcc14ebfe7b"
    if not openrouter_key or not elevenlabs_key:
        print("Both API keys are required.")
        return
        
    print("Initializing systems...")
    
    try:
        rag = RAGLayer(openrouter_key)
        voice_agent = VoiceAgent(elevenlabs_key, rag)
        print("Systems initialized successfully!")
        
        print("\n===== Romana Restaurant - Enhanced Voice Agent =====")
        print("Say 'exit' to quit, 'help' for commands, or 'type mode' to switch to keyboard input")
        
        conversation_history = []
        recognizer = sr.Recognizer()
        
        # Configure speech recognition for better reliability
        recognizer.energy_threshold = 3000  # Increase energy threshold
        recognizer.dynamic_energy_threshold = True  # Adapt dynamically to noise levels
        recognizer.pause_threshold = 1.0  # How long to wait for a pause before completing processing
        
        # Flag to switch between voice and text input modes
        voice_mode = True
        
        while True:
            query = ""
            
            if voice_mode:
                print("\nPlease speak your request (or say 'type mode' to switch to keyboard)...")
                # Initial pause to make sure speech recognition is ready
                time.sleep(0.5)
                
                try:
                    with sr.Microphone() as source:
                        print("Listening...")
                        # Adjust for ambient noise for 1 second
                        recognizer.adjust_for_ambient_noise(source, duration=1)
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        print("Processing audio...")
                    
                    try:
                        query = recognizer.recognize_google(audio)
                        print(f"You said: {query}")
                        
                        # Check for mode switch command
                        if "type mode" in query.lower():
                            voice_mode = False
                            print("Switching to keyboard input mode.")
                            continue
                    except sr.UnknownValueError:
                        print("Sorry, I could not understand the audio. Please try again or type 'type mode'.")
                        continue
                    except sr.RequestError as e:
                        print(f"Could not request results from Google Speech Recognition service; {e}")
                        voice_mode = False
                        print("Switching to keyboard input mode due to speech recognition error.")
                        continue
                except Exception as e:
                    print(f"Error with microphone: {e}")
                    voice_mode = False
                    print("Switching to keyboard input mode due to microphone error.")
                    continue
            else:
                # Text input mode
                query = input("\nType your request (or type 'voice mode' to switch back to speech): ")
                
                # Check for mode switch command
                if query.lower() == 'voice mode':
                    voice_mode = True
                    print("Switching to voice input mode.")
                    continue
            
            # Process exit command
            if query.lower() == 'exit':
                print("Thank you for using Romana Restaurant Voice Assistant. Goodbye!")
                break
            
            # Process help command
            elif query.lower() == 'help':
                help_text = voice_agent._get_available_commands()
                print("\n" + help_text)
                print("\nAdditional commands:")
                print("- 'exit' to quit the application")
                print("- 'voice mode' to use speech input")
                print("- 'type mode' to use keyboard input")
            else:
                print("Processing your request...")
                try:
                    # Add user message to history
                    conversation_history.append({"role": "user", "content": query})
                    
                    # Handle conversation
                    text, audio, conversation_history = voice_agent.handle_conversation(
                        query, conversation_history
                    )
                    
                    print(f"\nAssistant: {text}")
                    
                    if audio:
                        if voice_mode:
                            print("Playing audio response...")
                            voice_agent.play_audio(audio)
                        else:
                            print("(Audio response available in voice mode)")
                    else:
                        print("No audio generated")
                except Exception as e:
                    print(f"Error processing request: {str(e)}")
                    print("Try saying your request again or type 'help' for assistance.")
                
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        print("Please make sure your API keys are correct and all required libraries are installed.")

if __name__ == "__main__":
    interactive_demo()


