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






# import requests
# import json
# import os
# import logging
# import sounddevice as sd
# import soundfile as sf
# import io
# import numpy as np
# import re
# from typing import Optional, Dict, Any, List, Tuple
# from datetime import datetime, timedelta
# from rag_layer import RAGLayer
# import speech_recognition as sr
# import langdetect
# from langdetect import detect
# from googletrans import Translator

# class VoiceAgent:
#     """
#     Enhanced Voice Agent for Romana Restaurant with complete reservation,
#     ordering, and customer service capabilities.
#     Supports both English and Hindi languages.
#     """

#     def __init__(
#         self, 
#         elevenlabs_api_key: str,
#         dubverse_api_key: str,
#         rag_layer: RAGLayer,
#         voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel's voice
#         hindi_voice_id: str = "female",  # Default Dubverse female voice
#         stability: float = 0.5,
#         similarity_boost: float = 0.75,
#         audio_device: Optional[int] = None,
#         default_language: str = "en"  # Default language is English
#     ):
#         self.elevenlabs_api_key = elevenlabs_api_key
#         self.dubverse_api_key = dubverse_api_key
#         self.rag_layer = rag_layer
#         self.base_url = "https://api.elevenlabs.io/v1"
#         self.dubverse_url = "https://api.dubverse.ai/v1/speak"
#         self.audio_device = audio_device
#         self.default_language = default_language
#         self.translator = Translator()
        
#         # Voice configuration
#         self.voice_settings = {
#             "voice_id": voice_id,
#             "stability": stability,
#             "similarity_boost": similarity_boost,
#             "model_id": "eleven_monolingual_v1"
#         }
        
#         # Hindi voice configuration
#         self.hindi_voice_settings = {
#             "voice_id": hindi_voice_id
#         }
        
#         # Restaurant operational data
#         self.reservations = []
#         self.orders = []
#         self.current_order = None
#         self.current_reservation = None
#         self.feedback_records = []
        
#         self.logger = self._setup_logging()
        
#         # Restaurant configuration 
#         self.operating_hours = {
#             "Monday": "11:00 AM - 10:00 PM",
#             "Tuesday": "11:00 AM - 10:00 PM",
#             "Wednesday": "11:00 AM - 10:00 PM",
#             "Thursday": "11:00 AM - 10:00 PM",
#             "Friday": "11:00 AM - 11:00 PM",
#             "Saturday": "10:00 AM - 11:00 PM",
#             "Sunday": "10:00 AM - 10:00 PM"
#         }
        
#         # Hindi translation of restaurant configuration
#         self.operating_hours_hindi = {
#             "सोमवार": "सुबह 11:00 - रात 10:00",
#             "मंगलवार": "सुबह 11:00 - रात 10:00", 
#             "बुधवार": "सुबह 11:00 - रात 10:00",
#             "गुरुवार": "सुबह 11:00 - रात 10:00",
#             "शुक्रवार": "सुबह 11:00 - रात 11:00",
#             "शनिवार": "सुबह 10:00 - रात 11:00",
#             "रविवार": "सुबह 10:00 - रात 10:00"
#         }
        
#         self._validate_api_connection()
#         self._list_audio_devices()

#     def _start_ordering(self, query: str, conversation_history: List[Dict], language: str = "en") -> Tuple[str, bytes, List[Dict]]:
#         """Initiate food ordering process in appropriate language."""
#         self.current_order = {
#             "completed": False,
#             "step": "table_number",
#             "items": [],
#             "special_requests": "",
#             "language": language  # Store language preference
#         }
        
#         if language == "hi":
#             response = "रोमाना रेस्तरां में आपका स्वागत है! कृपया मुझे अपनी टेबल का नंबर बताएं ताकि मैं आपका ऑर्डर शुरू कर सकूं।"
#         else:
#             response = "Welcome to Romana Restaurant! Please tell me your table number so I can start your order."
        
#         audio = self.text_to_speech(response, language)
#         conversation_history.append({"role": "assistant", "content": response})
#         return response, audio, conversation_history

#     def _get_menu(self, language: str = "en") -> Dict:
#         """Retrieve restaurant menu from knowledge base and translate if needed."""
#         # This would normally come from a database or external source
#         menu = {
#             "appetizers": [
#                 {"name": "Bruschetta", "price": 8.99, "description": "Toasted bread topped with tomatoes, garlic, and basil"},
#                 {"name": "Caprese Salad", "price": 10.99, "description": "Fresh mozzarella, tomatoes, and basil drizzled with balsamic glaze"},
#                 {"name": "Garlic Bread", "price": 5.99, "description": "Freshly baked bread with garlic butter and herbs"}
#             ],
#             "main_courses": [
#                 {"name": "Margherita Pizza", "price": 14.99, "description": "Classic pizza with tomato sauce, mozzarella, and basil"},
#                 {"name": "Spaghetti Carbonara", "price": 16.99, "description": "Pasta with eggs, cheese, pancetta, and black pepper"},
#                 {"name": "Chicken Parmesan", "price": 18.99, "description": "Breaded chicken topped with marinara sauce and melted cheese"}
#             ],
#             "desserts": [
#                 {"name": "Tiramisu", "price": 7.99, "description": "Coffee-flavored Italian dessert with layers of mascarpone"},
#                 {"name": "Cannoli", "price": 6.99, "description": "Tube-shaped shells filled with sweet ricotta cream"},
#                 {"name": "Gelato", "price": 5.99, "description": "Italian ice cream in various flavors"}
#             ],
#             "beverages": [
#                 {"name": "Italian Soda", "price": 3.99, "description": "Carbonated water with flavored syrup"},
#                 {"name": "Espresso", "price": 2.99, "description": "Strong black coffee brewed by forcing steam through ground coffee beans"},
#                 {"name": "House Wine", "price": 8.99, "description": "Red or white wine selected by our sommelier"}
#             ]
#         }
        
#         # If language is Hindi, translate menu items
#         if language == "hi":
#             menu_hindi = {
#                 "स्टार्टर": [],
#                 "मुख्य व्यंजन": [],
#                 "मिठाई": [],
#                 "पेय पदार्थ": []
#             }
            
#             # Translate appetizers
#             for item in menu["appetizers"]:
#                 menu_hindi["स्टार्टर"].append({
#                     "name": self.translate_text(item["name"], "en", "hi"),
#                     "price": item["price"],
#                     "description": self.translate_text(item["description"], "en", "hi")
#                 })
            
#             # Translate main courses
#             for item in menu["main_courses"]:
#                 menu_hindi["मुख्य व्यंजन"].append({
#                     "name": self.translate_text(item["name"], "en", "hi"),
#                     "price": item["price"],
#                     "description": self.translate_text(item["description"], "en", "hi")
#                 })
            
#             # Translate desserts
#             for item in menu["desserts"]:
#                 menu_hindi["मिठाई"].append({
#                     "name": self.translate_text(item["name"], "en", "hi"),
#                     "price": item["price"],
#                     "description": self.translate_text(item["description"], "en", "hi")
#                 })
            
#             # Translate beverages
#             for item in menu["beverages"]:
#                 menu_hindi["पेय पदार्थ"].append({
#                     "name": self.translate_text(item["name"], "en", "hi"),
#                     "price": item["price"],
#                     "description": self.translate_text(item["description"], "en", "hi")
#                 })
            
#             return menu_hindi
        
#         return menu

#     def _validate_api_connection(self):
#         """Validate the connection to ElevenLabs API and Dubverse API."""
#         try:
#             # Validate ElevenLabs API
#             headers = {"xi-api-key": self.elevenlabs_api_key}
#             response = requests.get(f"{self.base_url}/voices", headers=headers)
#             if response.status_code != 200:
#                 raise RuntimeError(f"ElevenLabs API connection failed: {response.status_code}")
            
#             # Validate Dubverse API (just a basic check)
#             if not self.dubverse_api_key:
#                 raise RuntimeError("Dubverse API key is missing")
                
#             self.logger.info("API connections validated successfully")
            
#             # Call next validation
#             self._list_audio_devices()
#         except Exception as error:
#             self.logger.error(f"API validation failed: {str(error)}")
#             raise RuntimeError("Could not validate API connections")

#     def _list_audio_devices(self):
#         """List available audio devices and verify audio configuration."""
#         try:
#             sd.query_devices()
#             self.logger.info("Audio devices initialized successfully")
#         except Exception as error:
#             self.logger.error(f"Audio device initialization failed: {str(error)}")
#             raise RuntimeError("Could not initialize audio devices")

#     def _setup_logging(self):
#         """Configure logging system."""
#         os.makedirs("logs", exist_ok=True)
#         logger = logging.getLogger("VoiceAgent")
#         logger.setLevel(logging.INFO)
        
#         if not logger.handlers:
#             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
#             file_handler = logging.FileHandler(f"logs/voice_agent_{datetime.now().strftime('%Y%m%d')}.log")
#             file_handler.setFormatter(formatter)
            
#             console_handler = logging.StreamHandler()
#             console_handler.setLevel(logging.WARNING)
#             console_handler.setFormatter(formatter)
            
#             logger.addHandler(file_handler)
#             logger.addHandler(console_handler)
        
#         return logger

#     def detect_language(self, text: str) -> str:
#         """Detect the language of input text."""
#         try:
#             lang = detect(text)
#             if lang == "hi":
#                 return "hi"  # Hindi
#             return "en"  # Default to English
#         except langdetect.lang_detect_exception.LangDetectException:
#             return "en"  # Default to English if detection fails

#     def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
#         """Translate text between English and Hindi."""
#         if source_lang == target_lang:
#             return text
            
#         try:
#             translated = self.translator.translate(text, src=source_lang, dest=target_lang)
#             return translated.text
#         except Exception as e:
#             self.logger.error(f"Translation error: {str(e)}")
#             # Return original text if translation fails
#             return text

#     def text_to_speech(self, text: str, language: str = "en", output_file: Optional[str] = None) -> Optional[bytes]:
#         """
#         Convert text to speech using appropriate API based on language.
#         English: ElevenLabs API
#         Hindi: Dubverse API
#         """
#         if language == "hi":
#             return self._hindi_text_to_speech(text, output_file)
#         else:
#             return self._english_text_to_speech(text, output_file)

#     def _english_text_to_speech(self, text: str, output_file: Optional[str] = None) -> Optional[bytes]:
#         """Convert English text to speech using ElevenLabs API."""
#         try:
#             headers = {
#                 "xi-api-key": self.elevenlabs_api_key,
#                 "Content-Type": "application/json",
#                 "accept": "audio/mpeg"
#             }
            
#             data = {
#                 "text": text,
#                 "voice_settings": {
#                     "stability": self.voice_settings["stability"],
#                     "similarity_boost": self.voice_settings["similarity_boost"]
#                 },
#                 "model_id": self.voice_settings["model_id"]
#             }
            
#             response = requests.post(
#                 f"{self.base_url}/text-to-speech/{self.voice_settings['voice_id']}",
#                 headers=headers,
#                 json=data,
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 audio_data = response.content
#                 if output_file:
#                     with open(output_file, 'wb') as f:
#                         f.write(audio_data)
#                     return None
#                 return audio_data
#             else:
#                 raise RuntimeError(f"TTS failed: {response.status_code} - {response.text}")
                
#         except Exception as e:
#             self.logger.error(f"Error in English TTS conversion: {str(e)}")
#             raise RuntimeError(f"Error in English TTS conversion: {str(e)}")

#     def _hindi_text_to_speech(self, text: str, output_file: Optional[str] = None) -> Optional[bytes]:
#         """Convert Hindi text to speech using Dubverse API."""
#         try:
#             headers = {
#                 "Authorization": f"Bearer {self.dubverse_api_key}",
#                 "Content-Type": "application/json"
#             }
            
#             data = {
#                 "text": text,
#                 "voice": self.hindi_voice_settings["voice_id"],
#                 "language": "hi-IN"  # Hindi-India
#             }
            
#             response = requests.post(
#                 self.dubverse_url,
#                 headers=headers,
#                 json=data,
#                 timeout=60  # Longer timeout for Hindi TTS
#             )
            
#             if response.status_code == 200:
#                 response_json = response.json()
#                 # Get audio URL from response
#                 if "audio_url" in response_json:
#                     audio_url = response_json["audio_url"]
#                     # Download the audio file
#                     audio_response = requests.get(audio_url)
#                     if audio_response.status_code == 200:
#                         audio_data = audio_response.content
#                         if output_file:
#                             with open(output_file, 'wb') as f:
#                                 f.write(audio_data)
#                             return None
#                         return audio_data
#                     else:
#                         raise RuntimeError(f"Failed to download Hindi audio: {audio_response.status_code}")
#                 else:
#                     raise RuntimeError("No audio URL in Dubverse API response")
#             else:
#                 raise RuntimeError(f"Hindi TTS failed: {response.status_code} - {response.text}")
                
#         except Exception as e:
#             self.logger.error(f"Error in Hindi TTS conversion: {str(e)}")
#             raise RuntimeError(f"Error in Hindi TTS conversion: {str(e)}")

#     def play_audio(self, audio_data: bytes) -> None:
#         """Play audio from bytes."""
#         try:
#             audio_file = io.BytesIO(audio_data)
#             data, samplerate = sf.read(audio_file)
#             if len(data.shape) == 1:
#                 data = np.column_stack((data, data))
#             sd.play(data, samplerate, device=self.audio_device)
#             sd.wait()
#         except Exception as e:
#             self.logger.error(f"Error playing audio: {str(e)}")
#             raise RuntimeError(f"Error playing audio: {str(e)}")

#     # Enhanced Conversation Handling with Language Support

#     async def handle_conversation(self, query: str, conversation_history: List[Dict]) -> Tuple[str, Optional[bytes], List[Dict]]:
#         """
#         Main conversation handler that routes to specific functions based on context.
#         Automatically detects language and handles accordingly.
#         Returns tuple: (response_text, audio_data, updated_history)
#         """
#         try:
#             # Detect language of the query
#             detected_lang = self.detect_language(query)

#             # Check if we're in the middle of a reservation
#             if self.current_reservation and not self.current_reservation.get('completed', False):
#                 return self._handle_reservation_flow(query, conversation_history, detected_lang)

#             # Check if we're in the middle of an order
#             if self.current_order and not self.current_order.get('completed', False):
#                 return self._handle_ordering_flow(query, conversation_history, detected_lang)

#             # Normalize query for intent detection
#             query_lower = query.lower()

#             # Handle language switch requests
#             if "switch to english" in query_lower or "अंग्रेजी में बात करें" in query_lower:
#                 self.default_language = "en"
#                 response = "I've switched to English. How can I help you today?"
#                 if detected_lang == "hi":
#                     response = "मैंने अंग्रेजी में स्विच कर दिया है। मैं आपकी कैसे मदद कर सकता हूँ?"
#                 audio = self.text_to_speech(response, detected_lang)
#                 conversation_history.append({"role": "assistant", "content": response})
#                 return response, audio, conversation_history

#             if "switch to hindi" in query_lower or "हिंदी में बात करें" in query_lower:
#                 self.default_language = "hi"
#                 response = "I've switched to Hindi. How can I help you today?"
#                 response_hindi = "मैंने हिंदी में स्विच कर दिया है। मैं आपकी कैसे मदद कर सकता हूँ?"
#                 audio = await self.text_to_speech(response_hindi, "hi")
#                 conversation_history.append({"role": "assistant", "content": response_hindi})
#                 return response_hindi, audio, conversation_history

#             # Intent dictionaries...
#             english_intents = {
#                 "reservation": ["reservation", "book", "table", "reserve"],
#                 "order": ["order", "menu", "food", "dish", "eat", "hungry"],
#                 "hours": ["hours", "open", "close", "timing", "schedule"],
#                 "feedback": ["feedback", "review", "experience", "comment"],
#                 "help": ["help", "commands", "options", "what can you do"],
#                 "specials": ["specials", "today", "chef", "recommend", "popular"],
#                 "location": ["location", "address", "directions", "find"]
#             }

#             hindi_intents = {
#                 "reservation": ["आरक्षण", "बुकिंग", "टेबल", "रिजर्व", "बुक"],
#                 "order": ["ऑर्डर", "मेनू", "खाना", "भोजन", "खाद्य", "भूख"],
#                 "hours": ["समय", "खुलने का समय", "बंद होने का समय", "टाइमिंग", "शेड्यूल"],
#                 "feedback": ["प्रतिक्रिया", "समीक्षा", "अनुभव", "टिप्पणी"],
#                 "help": ["मदद", "सहायता", "कमांड", "विकल्प", "आप क्या कर सकते हैं"],
#                 "specials": ["विशेष", "आज", "शेफ", "अनुशंसा", "लोकप्रिय"],
#                 "location": ["स्थान", "पता", "दिशा-निर्देश", "ढूंढना"]
#             }

#             intents = english_intents if detected_lang == "en" else hindi_intents

#             for intent, keywords in intents.items():
#                 if any(word in query_lower for word in keywords):
#                     if intent == "reservation":
#                         return self._start_reservation(query, conversation_history, detected_lang)
#                     elif intent == "order":
#                         return await self._start_ordering(query, conversation_history, detected_lang)
#                     elif intent == "hours":
#                         response = self._get_operating_hours(detected_lang)
#                         audio = await self.text_to_speech(response, detected_lang)
#                         conversation_history.append({"role": "assistant", "content": response})
#                         return response, audio, conversation_history
#                     elif intent == "feedback":
#                         return await self._handle_feedback(query, conversation_history, detected_lang)
#                     elif intent == "help":
#                         response = self._get_available_commands(detected_lang)
#                         audio = await self.text_to_speech(response, detected_lang)
#                         conversation_history.append({"role": "assistant", "content": response})
#                         return response, audio, conversation_history
#                     elif intent == "specials":
#                         return await self._get_daily_specials(conversation_history, detected_lang)
#                     elif intent == "location":
#                         return await self._get_location_info(conversation_history, detected_lang)

#             # No intent matched — fallback to RAG
#             rag_query = query
#             if detected_lang == "hi":
#                 rag_query = self.translate_text(query, "hi", "en")

#             rag_response, rag_audio, updated_history = self.rag_layer.generate_response(rag_query, conversation_history)

#             if detected_lang == "hi":
#                 hindi_response = self.translate_text(rag_response, "en", "hi")
#                 hindi_audio = self.text_to_speech(hindi_response, "hi")
#                 updated_history[-1]["content"] = hindi_response
#                 return hindi_response, hindi_audio, updated_history

#             return rag_response, rag_audio, updated_history

#         except Exception as e:
#             self.logger.error(f"Error in conversation handling: {str(e)}")
#             if 'detected_lang' not in locals():
#                 detected_lang = "en"
#             error_msg = "I'm sorry, I encountered an error processing your request. Could you please try again?" if detected_lang == "en" else "मुझे क्षमा करें, आपके अनुरोध को संसाधित करने में एक त्रुटि हुई। कृपया फिर से प्रयास करें।"
#             try:
#                 audio = self.text_to_speech(error_msg, detected_lang)
#                 return error_msg, audio, conversation_history
#             except:
#                 return error_msg, None, conversation_history



#     # Reservation System with Language Support
#     def _start_reservation(self, query: str, conversation_history: List[Dict], language: str = "en") -> Tuple[str, bytes, List[Dict]]:
#         """Initiate reservation process in appropriate language."""
#         self.current_reservation = {
#             "completed": False,
#             "step": "party_size",
#             "data": {},
#             "language": language  # Store the language preference
#         }
        
#         if language == "hi":
#             response = "रोमाना रेस्तरां चुनने के लिए धन्यवाद! कृपया मुझे बताएं कि कितने लोग खाना खाएंगे? बस एक संख्या बताएं।"
#         else:
#             response = "Thank you for choosing Romana Restaurant! Please tell me how many people will be dining with us? Just say a number."
        
#         audio = self.text_to_speech(response, language)
#         conversation_history.append({"role": "assistant", "content": response})
#         return response, audio, conversation_history

#     def _handle_reservation_flow(self, query: str, conversation_history: List[Dict], language: str = "en") -> Tuple[str, bytes, List[Dict]]:
#         """Handle multi-step reservation process with language support."""
#         # Use the language stored in the reservation object, or the current detected language
#         res_language = self.current_reservation.get("language", language)
#         current_step = self.current_reservation["step"]
        
#         if current_step == "party_size":
#             # Extract number from text
#             party_size = self._extract_number_from_text(query)
            
#             if party_size is not None and party_size > 0:
#                 self.current_reservation["data"]["party_size"] = party_size
#                 self.current_reservation["step"] = "date"
                
#                 if res_language == "hi":
#                     response = f"बहुत अच्छा! हम {party_size} लोगों के लिए आरक्षण करेंगे। आप किस तारीख को हमारे साथ भोजन करना चाहेंगे? आप कल, शुक्रवार या किसी विशेष तारीख को कह सकते हैं।"
#                 else:
#                     response = f"Great! We'll reserve for {party_size} people. What date would you like to dine with us? You can say tomorrow, Friday, or a specific date."
                
#                 audio = self.text_to_speech(response, res_language)
#                 conversation_history.append({"role": "assistant", "content": response})
#                 return response, audio, conversation_history
#             else:
#                 if res_language == "hi":
#                     response = "मुझे जानना होगा कि कितने लोग खाना खाएंगे। कृपया केवल एक संख्या बताएं, जैसे 'चार' या 'छह'।"
#                 else:
#                     response = "I need to know how many people will be dining. Please say just a number, like 'four' or 'six'."
                
#                 audio = self.text_to_speech(response, res_language)
#                 return response, audio, conversation_history
        
#         elif current_step == "date":
#             try:
#                 parsed_date = self._parse_date(query)
#                 if parsed_date < datetime.now().date():
#                     if res_language == "hi":
#                         response = "मुझे क्षमा करें, हम अतीत की तारीखों के लिए आरक्षण नहीं कर सकते। कृपया भविष्य की तारीख चुनें।"
#                     else:
#                         response = "I'm sorry, we can't make reservations for dates in the past. Please choose a future date."
                    
#                     audio = self.text_to_speech(response, res_language)
#                     return response, audio, conversation_history
                
#                 self.current_reservation["data"]["date"] = parsed_date.strftime("%Y-%m-%d")
#                 self.current_reservation["step"] = "time"
                
#                 if res_language == "hi":
#                     response = "आप किस समय आरक्षण करना चाहेंगे? हमारा समय सुबह 11 बजे से रात 10 बजे तक है।"
#                 else:
#                     response = "What time would you like to reserve? Our hours are 11AM to 10PM."
                
#                 audio = self.text_to_speech(response, res_language)
#                 conversation_history.append({"role": "assistant", "content": response})
#                 return response, audio, conversation_history
                
#             except Exception as e:
#                 if res_language == "hi":
#                     response = "मैं उस तारीख को समझ नहीं पाया। कृपया 'कल', 'इस शुक्रवार', या '20 मई' जैसा कुछ कहें।"
#                 else:
#                     response = "I couldn't understand that date. Please say something like 'tomorrow', 'this Friday', or 'May 20th'."
                
#                 audio = self.text_to_speech(response, res_language)
#                 return response, audio, conversation_history
        
#         elif current_step == "time":
#             try:
#                 parsed_time = self._parse_time(query)
#                 self.current_reservation["data"]["time"] = parsed_time.strftime("%I:%M %p")
#                 self.current_reservation["step"] = "name"
                
#                 if res_language == "hi":
#                     response = "बिल्कुल सही! आरक्षण किस नाम से करना चाहेंगे?"
#                 else:
#                     response = "Perfect! What name should I put the reservation under?"
                
#                 audio = self.text_to_speech(response, res_language)
#                 conversation_history.append({"role": "assistant", "content": response})
#                 return response, audio, conversation_history
                
#             except Exception as e:
#                 if res_language == "hi":
#                     response = "कृपया सुबह 11 बजे और रात 10 बजे के बीच एक वैध समय बताएं, जैसे 'शाम सात बजकर तीस मिनट' या 'दोपहर 12:45'।"
#                 else:
#                     response = "Please tell me a valid time between 11AM and 10PM, like 'seven thirty PM' or '12:45 PM'."
                
#                 audio = self.text_to_speech(response, res_language)
#                 return response, audio, conversation_history
        
#         elif current_step == "name":
#             self.current_reservation["data"]["name"] = query
#             self.current_reservation["step"] = "phone"
            
#             if res_language == "hi":
#                 response = "धन्यवाद। क्या मुझे एक संपर्क फोन नंबर भी मिल सकता है, अगर हमें आपसे संपर्क करने की आवश्यकता हो?"
#             else:
#                 response = "Thank you. Could I also have a contact phone number in case we need to reach you?"
            
#             audio = self.text_to_speech(response, res_language)
#             conversation_history.append({"role": "assistant", "content": response})
#             return response, audio, conversation_history
            
#         elif current_step == "phone":
#             # Simple validation - we're just checking if there are digits
#             if any(char.isdigit() for char in query):
#                 self.current_reservation["data"]["phone"] = query
#                 self.current_reservation["step"] = "confirm"
                
#                 # Format confirmation message
#                 res_data = self.current_reservation["data"]
                
#                 if res_language == "hi":
#                     response = (
#                         f"मुझे आपके आरक्षण की पुष्टि करने दें:\n"
#                         f"नाम: {res_data['name']}\n"
#                         f"फोन: {res_data['phone']}\n"
#                         f"पार्टी का आकार: {res_data['party_size']}\n"
#                         f"तारीख: {res_data['date']}\n"
#                         f"समय: {res_data['time']}\n\n"
#                         f"क्या यह जानकारी सही है? कृपया हां या नहीं कहें।"
#                     )
#                 else:
#                     response = (
#                         f"Let me confirm your reservation:\n"
#                         f"Name: {res_data['name']}\n"
#                         f"Phone: {res_data['phone']}\n"
#                         f"Party Size: {res_data['party_size']}\n"
#                         f"Date: {res_data['date']}\n"
#                         f"Time: {res_data['time']}\n\n"
#                         f"Is this information correct? Please say yes or no."
#                     )
                
#                 audio = self.text_to_speech(response, res_language)
#                 conversation_history.append({"role": "assistant", "content": response})
#                 return response, audio, conversation_history
#             else:
#                 if res_language == "hi":
#                     response = "मुझे अंकों वाला फोन नंबर चाहिए। कृपया एक वैध फोन नंबर प्रदान करें।"
#                 else:
#                     response = "I need a phone number with digits. Please provide a valid phone number."
                
#                 audio = self.text_to_speech(response, res_language)
#                 return response, audio, conversation_history
        
#         elif current_step == "confirm":
#             # Check for confirmation in appropriate language
#             confirmation_words = {"yes", "correct", "right"} if res_language == "en" else {"हां", "सही", "ठीक"}
            
#             if any(word in query.lower() for word in confirmation_words):
#                 # Complete reservation
#                 self.reservations.append(self.current_reservation["data"])
#                 self.current_reservation["completed"] = True
                
#                 if res_language == "hi":
#                     response = (
#                         "आपका आरक्षण पुष्ट हो गया है! हम रोमाना रेस्तरां में आपकी सेवा करने के लिए तत्पर हैं। "
#                         "क्या आपके पास कोई विशेष अनुरोध या आहार संबंधी प्रतिबंध हैं जिन्हें हमें जानना चाहिए?"
#                     )
#                 else:
#                     response = (
#                         "Your reservation is confirmed! We look forward to serving you at Romana Restaurant. "
#                         "Do you have any special requests or dietary restrictions we should know about?"
#                     )
                
#                 audio = self.text_to_speech(response, res_language)
#                 conversation_history.append({"role": "assistant", "content": response})
#                 return response, audio, conversation_history
#             else:
#                 self.current_reservation["step"] = "party_size"
                
#                 if res_language == "hi":
#                     response = "फिर से शुरू करते हैं। कितने लोग हमारे साथ खाना खाएंगे?"
#                 else:
#                     response = "Let's start over. How many people will be dining with us?"
                
#                 audio = self.text_to_speech(response, res_language)
#                 return response, audio, conversation_history

#     # Ordering System with Language Support
#     def _handle_ordering_flow(self, query: str, conversation_history: List[Dict], language: str = "en") -> Tuple[str, bytes, List[Dict]]:
#         """Handle multi-step food ordering process with language support."""
#         # Use the language stored in the order object, or the current detected language
#         order_language = self.current_order.get("language", language)
#         current_step = self.current_order["step"]
        
#         if current_step == "table_number":
#             # Extract number from text
#             table_number = self._extract_number_from_text(query)
            
#             if table_number is not None and table_number > 0:
#                 self.current_order["table_number"] = table_number
#                 self.current_order["step"] = "food_selection"
                
#                 # Get menu and format it for presentation
#                 menu = self._get_menu(order_language)
                
#                 if order_language == "hi":
#                     response = f"धन्यवाद! टेबल {table_number} के लिए आपका ऑर्डर शुरू हो गया है।\n\n"
#                     response += "हमारा मेनू यहां है:\n\n"
                    
#                     # Format Hindi menu
#                     for category, items in menu.items():
#                         response += f"🍽️ {category}:\n"
#                         for item in items:
#                             response += f"  • {item['name']} - ₹{item['price']:.2f}\n"
#                             response += f"    {item['description']}\n"
#                         response += "\n"
                    
#                     response += "आप क्या ऑर्डर करना चाहेंगे? आप कह सकते हैं 'मार्गरीटा पिज्जा' या 'एक एस्प्रेसो'।"
#                 else:
#                     response = f"Thank you! Your order for table {table_number} has begun.\n\n"
#                     response += "Here's our menu:\n\n"
                    
#                     # Format English menu
#                     for category, items in menu.items():
#                         response += f"🍽️ {category.title()}:\n"
#                         for item in items:
#                             response += f"  • {item['name']} - ${item['price']:.2f}\n"
#                             response += f"    {item['description']}\n"
#                         response += "\n"
                    
#                     response += "What would you like to order? You can say 'Margherita Pizza' or 'an Espresso'."
                
#                 audio = self.text_to_speech(response, order_language)
#                 conversation_history.append({"role": "assistant", "content": response})
#                 return response, audio, conversation_history
#             else:
#                 if order_language == "hi":
#                     response = "कृपया मुझे एक वैध टेबल नंबर बताएं, जैसे '5' या 'टेबल 3'।"
#                 else:
#                     response = "Please provide a valid table number, like '5' or 'table 3'."
                
#                 audio = self.text_to_speech(response, order_language)
#                 return response, audio, conversation_history
        
#         elif current_step == "food_selection":
#             # Check if the user is done ordering
#             done_keywords_en = ["that's all", "finished", "complete", "done ordering"]
#             done_keywords_hi = ["बस इतना ही", "पूरा हो गया", "हो गया", "ऑर्डर पूरा"]
            
#             done_keywords = done_keywords_en if order_language == "en" else done_keywords_hi
            
#             if any(keyword in query.lower() for keyword in done_keywords):
#                 self.current_order["step"] = "confirm_order"
                
#                 # Show order summary
#                 if order_language == "hi":
#                     response = "आपका ऑर्डर:\n"
#                     total = 0
#                     for item in self.current_order["items"]:
#                         response += f"• {item['name']} - ₹{item['price']:.2f}\n"
#                         total += item['price']
                    
#                     response += f"\nकुल: ₹{total:.2f}\n"
#                     response += "क्या आप इस ऑर्डर की पुष्टि करना चाहते हैं? कृपया हां या नहीं कहें।"
#                 else:
#                     response = "Your order:\n"
#                     total = 0
#                     for item in self.current_order["items"]:
#                         response += f"• {item['name']} - ${item['price']:.2f}\n"
#                         total += item['price']
                    
#                     response += f"\nTotal: ${total:.2f}\n"
#                     response += "Would you like to confirm this order? Please say yes or no."
                
#                 audio = self.text_to_speech(response, order_language)
#                 conversation_history.append({"role": "assistant", "content": response})
#                 return response, audio, conversation_history
            
#             # Try to match the order item to our menu
#             menu = self._get_menu(order_language)
#             found_item = None
            
#             # Flatten menu for easier searching
#             all_items = []
#             for category, items in menu.items():
#                 all_items.extend(items)
            
#             # Search for matching item in menu
#             query_normalized = query.lower()
#             for item in all_items:
#                 if item["name"].lower() in query_normalized:
#                     found_item = item
#                     break
            
#             if found_item:
#                 # Add item to order
#                 self.current_order["items"].append(found_item)
                
#                 if order_language == "hi":
#                     response = f"{found_item['name']} आपके ऑर्डर में जोड़ दिया गया है। क्या आप कुछ और ऑर्डर करना चाहेंगे? या 'बस इतना ही' कहें।"
#                 else:
#                     response = f"{found_item['name']} has been added to your order. Would you like to order anything else? Or say 'that's all'."
                
#                 audio = self.text_to_speech(response, order_language)
#                 conversation_history.append({"role": "assistant", "content": response})
#                 return response, audio, conversation_history
#             else:
#                 if order_language == "hi":
#                     response = "मुझे क्षमा करें, मैं उस आइटम को मेनू में नहीं पा सका। कृपया फिर से प्रयास करें या किसी अन्य आइटम का नाम बताएं।"
#                 else:
#                     response = "I'm sorry, I couldn't find that item on our menu. Please try again or specify a different item."
                
#                 audio = self.text_to_speech(response, order_language)
#                 return response, audio, conversation_history
        
#         elif current_step == "confirm_order":
#             # Check for confirmation in appropriate language
#             confirmation_words = {"yes", "correct", "right", "confirm"} if order_language == "en" else {"हां", "सही", "ठीक", "पुष्टि"}
            
#             if any(word in query.lower() for word in confirmation_words):
#                 # Ask for special requests
#                 self.current_order["step"] = "special_requests"
                
#                 if order_language == "hi":
#                     response = "क्या आपके पास कोई विशेष अनुरोध है? जैसे कि 'बिना प्याज के' या 'अतिरिक्त पनीर'। या 'कोई नहीं' कहें।"
#                 else:
#                     response = "Do you have any special requests? Like 'no onions' or 'extra cheese'. Or say 'none'."
                
#                 audio = self.text_to_speech(response, order_language)
#                 conversation_history.append({"role": "assistant", "content": response})
#                 return response, audio, conversation_history
#             else:
#                 # Start over with food selection
#                 self.current_order["items"] = []
#                 self.current_order["step"] = "food_selection"
                
#                 if order_language == "hi":
#                     response = "फिर से शुरू करते हैं। आप क्या ऑर्डर करना चाहेंगे?"
#                 else:
#                     response = "Let's start over. What would you like to order?"
                
#                 audio = self.text_to_speech(response, order_language)
#                 return response, audio, conversation_history
        
#         elif current_step == "special_requests":
#             # Check if no special requests
#             none_keywords_en = ["none", "no", "nothing special"]
#             none_keywords_hi = ["कुछ नहीं", "नहीं", "कोई नहीं"]
            
#             none_keywords = none_keywords_en if order_language == "en" else none_keywords_hi
            
#             if not any(keyword in query.lower() for keyword in none_keywords):
#                 self.current_order["special_requests"] = query
            
#             # Complete the order
#             self.orders.append({
#                 "table_number": self.current_order["table_number"],
#                 "items": self.current_order["items"],
#                 "special_requests": self.current_order["special_requests"],
#                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             })
            
#             self.current_order["completed"] = True
            
#             if order_language == "hi":
#                 response = "आपका ऑर्डर सफलतापूर्वक दर्ज किया गया है! हम जल्द ही टेबल पर आपका भोजन लेकर आएंगे। क्या मैं आपकी और कोई मदद कर सकता हूँ?"
#             else:
#                 response = "Your order has been successfully placed! We'll bring your food to your table shortly. Is there anything else I can help you with?"
            
#             audio = self.text_to_speech(response, order_language)
#             conversation_history.append({"role": "assistant", "content": response})
#             return response, audio, conversation_history

#     # Helper methods for parsing user input
#     def _extract_number_from_text(self, text: str) -> Optional[int]:
#         """Extract a numerical value from text in either language."""
#         # Try to find digits first
#         digits_match = re.search(r'\d+', text)
#         if digits_match:
#             return int(digits_match.group())
        
#         # Word to number mapping (English)
#         english_numbers = {
#             'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
#             'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
#         }
        
#         # Word to number mapping (Hindi)
#         hindi_numbers = {
#             'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4, 'पांच': 5,
#             'छह': 6, 'सात': 7, 'आठ': 8, 'नौ': 9, 'दस': 10
#         }
        
#         # Check for English number words
#         for word, number in english_numbers.items():
#             if word in text.lower():
#                 return number
        
#         # Check for Hindi number words
#         for word, number in hindi_numbers.items():
#             if word in text:
#                 return number
        
#         return None

#     def _parse_date(self, text: str) -> datetime.date:
#         """Parse date from text in either language."""
#         text_lower = text.lower()
#         today = datetime.now().date()
        
#         # Check for relative dates first (English)
#         if "today" in text_lower:
#             return today
#         elif "tomorrow" in text_lower:
#             return today + timedelta(days=1)
#         elif "day after tomorrow" in text_lower:
#             return today + timedelta(days=2)
        
#         # Check for Hindi relative dates
#         if "आज" in text_lower:
#             return today
#         elif "कल" in text_lower:
#             return today + timedelta(days=1)
#         elif "परसों" in text_lower:
#             return today + timedelta(days=2)
        
#         # Check for day names (English)
#         days_en = {
#             "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
#             "friday": 4, "saturday": 5, "sunday": 6
#         }
        
#         # Check for day names (Hindi)
#         days_hi = {
#             "सोमवार": 0, "मंगलवार": 1, "बुधवार": 2, "गुरुवार": 3,
#             "शुक्रवार": 4, "शनिवार": 5, "रविवार": 6
#         }
        
#         # Check English day names
#         for day_name, day_num in days_en.items():
#             if day_name in text_lower:
#                 current_day = today.weekday()
#                 days_ahead = (day_num - current_day) % 7
#                 if days_ahead == 0:
#                     days_ahead = 7  # If today, go to next week
#                 return today + timedelta(days=days_ahead)
        
#         # Check Hindi day names
#         for day_name, day_num in days_hi.items():
#             if day_name in text_lower:
#                 current_day = today.weekday()
#                 days_ahead = (day_num - current_day) % 7
#                 if days_ahead == 0:
#                     days_ahead = 7  # If today, go to next week
#                 return today + timedelta(days=days_ahead)
        
#         # Try to extract month and day numerically
#         date_pattern = r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
#         month_map = {
#             'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
#             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
#         }
        
#         match = re.search(date_pattern, text_lower)
#         if match:
#             day = int(match.group(1))
#             month_str = match.group(2)[:3]
#             month = month_map[month_str]
#             year = today.year
            
#             # If the month is before current month, assume next year
#             if month < today.month:
#                 year += 1
            
#             return datetime(year, month, day).date()
        
#         # If all else fails, default to tomorrow
#         return today + timedelta(days=1)

#     def _parse_time(self, text: str) -> datetime:
#         """Parse time from text in either language."""
#         text_lower = text.lower()
#         now = datetime.now()
        
#         # Try to extract time in HH:MM format with AM/PM
#         time_pattern = r'(\d{1,2}):?(\d{2})?\s*(am|pm)?'
#         match = re.search(time_pattern, text_lower)
        
#         if match:
#             hour = int(match.group(1))
#             minute = int(match.group(2) or 0)
#             ampm = match.group(3)
            
#             # Adjust for 12-hour format if AM/PM is specified
#             if ampm:
#                 if ampm == 'pm' and hour < 12:
#                     hour += 12
#                 elif ampm == 'am' and hour == 12:
#                     hour = 0
#             # If no AM/PM but hour < 12, assume PM for reasonable dining times
#             elif hour < 12 and hour >= 6:
#                 hour += 12
            
#             # Create datetime with today's date and parsed time
#             return datetime.combine(now.date(), datetime.min.time()).replace(hour=hour, minute=minute)
        
#         # Check for common time phrases in English
#         if "noon" in text_lower:
#             return datetime.combine(now.date(), datetime.min.time()).replace(hour=12, minute=0)
#         elif "midnight" in text_lower:
#             return datetime.combine(now.date(), datetime.min.time()).replace(hour=0, minute=0)
        
#         # Check for Hindi time references
#         if "दोपहर" in text_lower:  # Noon
#             return datetime.combine(now.date(), datetime.min.time()).replace(hour=12, minute=0)
#         elif "शाम" in text_lower:  # Evening
#             # Try to extract a number for evening time
#             evening_time = self._extract_number_from_text(text_lower)
#             if evening_time and evening_time < 12:
#                 return datetime.combine(now.date(), datetime.min.time()).replace(hour=evening_time + 12, minute=0)
#             else:
#                 return datetime.combine(now.date(), datetime.min.time()).replace(hour=18, minute=0)  # Default to 6 PM
#         elif "सुबह" in text_lower:  # Morning
#             morning_time = self._extract_number_from_text(text_lower)
#             if morning_time:
#                 return datetime.combine(now.date(), datetime.min.time()).replace(hour=morning_time, minute=0)
#             else:
#                 return datetime.combine(now.date(), datetime.min.time()).replace(hour=9, minute=0)  # Default to 9 AM
        
#         # Default to 7:00 PM if no time could be extracted
#         return datetime.combine(now.date(), datetime.min.time()).replace(hour=19, minute=0)

#     # Information Functions
#     def _get_operating_hours(self, language: str = "en") -> str:
#         """Return restaurant operating hours in the appropriate language."""
#         if language == "hi":
#             response = "रोमाना रेस्तरां का परिचालन समय:\n\n"
#             for day, hours in self.operating_hours_hindi.items():
#                 response += f"{day}: {hours}\n"
#         else:
#             response = "Romana Restaurant Operating Hours:\n\n"
#             for day, hours in self.operating_hours.items():
#                 response += f"{day}: {hours}\n"
        
#         return response

#     def _get_available_commands(self, language: str = "en") -> str:
#         """Return available commands and features in the appropriate language."""
#         if language == "hi":
#             return (
#                 "मैं आपकी इन तरीकों से मदद कर सकता हूँ:\n\n"
#                 "🍽️ टेबल आरक्षण करना\n"
#                 "🥗 भोजन ऑर्डर करना\n"
#                 "🕒 परिचालन समय देखना\n"
#                 "📝 प्रतिक्रिया देना\n"
#                 "👨‍🍳 आज के विशेष व्यंजन\n"
#                 "📍 रेस्तरां का स्थान और दिशा-निर्देश\n"
#                 "🔤 भाषा बदलने के लिए 'अंग्रेजी में बात करें' या 'हिंदी में बात करें' कहें\n\n"
#                 "बस अपने अनुरोध के बारे में बात करें, और मैं आपकी मदद करूंगा!"
#             )
#         else:
#             return (
#                 "I can help you in these ways:\n\n"
#                 "🍽️ Make a table reservation\n"
#                 "🥗 Order food\n"
#                 "🕒 Check operating hours\n"
#                 "📝 Provide feedback\n"
#                 "👨‍🍳 Today's specials\n"
#                 "📍 Restaurant location and directions\n"
#                 "🔤 Say 'switch to Hindi' or 'switch to English' to change languages\n\n"
#                 "Just talk about your request, and I'll help you out!"
#             )

#     def _get_daily_specials(self, conversation_history: List[Dict], language: str = "en") -> Tuple[str, bytes, List[Dict]]:
#         """Return information about daily specials in the appropriate language."""
#         # This would normally be retrieved from a database
#         specials = {
#             "en": {
#                 "starter": "Truffle Arancini - Risotto balls stuffed with mushrooms and truffle oil",
#                 "main": "Seafood Linguine - Fresh pasta with shrimp, clams, and mussels in a white wine sauce",
#                 "dessert": "Limoncello Panna Cotta - Creamy Italian dessert with a hint of lemon liqueur",
#                 "price": "$29.99"
#             },
#             "hi": {
#                 "starter": "ट्रफल आरंचिनी - मशरूम और ट्रफल तेल के साथ भरे हुए रिसोट्टो बॉल्स",
#                 "main": "सीफूड लिंगुइन - व्हाइट वाइन सॉस में झींगा, क्लैम्स और मसल्स के साथ ताजा पास्ता",
#                 "dessert": "लिमोनचेलो पन्ना कोट्टा - नींबू के लिकर के स्वाद के साथ क्रीमी इतालवी मिठाई",
#                 "price": "₹2,299"
#             }
#         }
        
#         special = specials["hi"] if language == "hi" else specials["en"]
        
#         if language == "hi":
#             response = (
#                 "🌟 आज का विशेष मेनू 🌟\n\n"
#                 f"स्टार्टर: {special['starter']}\n\n"
#                 f"मुख्य व्यंजन: {special['main']}\n\n"
#                 f"मिठाई: {special['dessert']}\n\n"
#                 f"कुल मूल्य: {special['price']} (पूरे सेट के लिए विशेष मूल्य)\n\n"
#                 "क्या आप इसे आज ऑर्डर करना चाहेंगे?"
#             )
#         else:
#             response = (
#                 "🌟 Today's Special Menu 🌟\n\n"
#                 f"Starter: {special['starter']}\n\n"
#                 f"Main Course: {special['main']}\n\n"
#                 f"Dessert: {special['dessert']}\n\n"
#                 f"Total Price: {special['price']} (special price for the complete set)\n\n"
#                 "Would you like to order this today?"
#             )
        
#         audio = self.text_to_speech(response, language)
#         conversation_history.append({"role": "assistant", "content": response})
#         return response, audio, conversation_history

#     def _get_location_info(self, conversation_history: List[Dict], language: str = "en") -> Tuple[str, bytes, List[Dict]]:
#         """Return restaurant location information in the appropriate language."""
#         location_info = {
#             "en": {
#                 "address": "123 Culinary Street, Gourmet District, Foodville",
#                 "landmarks": "Next to Central Park, opposite the Grand Library",
#                 "parking": "Free parking available in the restaurant's underground garage",
#                 "public_transport": "Bus routes 42, 67 stop directly in front. Metro Station 'Gourmet Square' is a 5-minute walk"
#             },
#             "hi": {
#                 "address": "123 कुलिनरी स्ट्रीट, गोरमेट जिला, फूडविल",
#                 "landmarks": "सेंट्रल पार्क के बगल में, ग्रैंड लाइब्रेरी के सामने",
#                 "parking": "रेस्तरां के भूमिगत गैरेज में मुफ्त पार्किंग उपलब्ध है",
#                 "public_transport": "बस मार्ग 42, 67 सीधे सामने रुकते हैं। मेट्रो स्टेशन 'गोरमेट स्क्वायर' 5 मिनट की पैदल दूरी पर है"
#             }
#         }
        
#         info = location_info["hi"] if language == "hi" else location_info["en"]
        
#         if language == "hi":
#             response = (
#                 "📍 रोमाना रेस्तरां का स्थान 📍\n\n"
#                 f"पता: {info['address']}\n\n"
#                 f"प्रमुख स्थान: {info['landmarks']}\n\n"
#                 f"पार्किंग: {info['parking']}\n\n"
#                 f"सार्वजनिक परिवहन: {info['public_transport']}\n\n"
#                 "क्या आप दिशा-निर्देशों के लिए हमारी वेबसाइट पर जाना चाहेंगे या एक आरक्षण करना चाहेंगे?"
#             )
#         else:
#             response = (
#                 "📍 Romana Restaurant Location 📍\n\n"
#                 f"Address: {info['address']}\n\n"
#                 f"Landmarks: {info['landmarks']}\n\n"
#                 f"Parking: {info['parking']}\n\n"
#                 f"Public Transport: {info['public_transport']}\n\n"
#                 "Would you like to visit our website for directions or make a reservation?"
#             )
        
#         audio = self.text_to_speech(response, language)
#         conversation_history.append({"role": "assistant", "content": response})
#         return response, audio, conversation_history

#     def _handle_feedback(self, query: str, conversation_history: List[Dict], language: str = "en") -> Tuple[str, bytes, List[Dict]]:
#         """Handle customer feedback in the appropriate language."""
#         # Save the feedback to our records
#         self.feedback_records.append({
#             "feedback": query,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "language": language
#         })
        
#         if language == "hi":
#                         response = (
#                 "आपकी प्रतिक्रिया के लिए धन्यवाद! हम अपनी सेवाओं को बेहतर बनाने के लिए आपके सुझावों को महत्व देते हैं। "
#                 "आपकी प्रतिक्रिया हमारे प्रबंधकों को भेज दी गई है। "
#                 "क्या आप हमारी सहायता से और कुछ जानना चाहेंगे?"
#             )
#         else:
#             response = (
#                 "Thank you for your feedback! We value your suggestions to improve our services. "
#                 "Your feedback has been forwarded to our managers. "
#                 "Is there anything else you would like to know about our services?"
#             )
        
#         audio = self.text_to_speech(response, language)
#         conversation_history.append({"role": "assistant", "content": response})
#         return response, audio, conversation_history



# import os
# import speech_recognition as sr
# import asyncio

# async def main():
#     # Replace with your actual API keys and RAGLayer instance
#     elevenlabs_api_key = "sk_a643471cf3d2de658ac47648b33d8314bfe39dcc14ebfe7b"
#     dubverse_api_key = "2qBhfJ5adu9wZSQkgTbnBonekIrNeoSK"
#     api_key = os.getenv("OPENROUTER_API_KEY") or input("Enter your API key: ")
#     rag_layer = RAGLayer(api_key)

#     # Initialize the restaurant assistant
#     assistant = VoiceAgent(
#         elevenlabs_api_key=elevenlabs_api_key,
#         dubverse_api_key=dubverse_api_key,
#         rag_layer=rag_layer
#     )

#     # Initialize the speech recognizer
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     print("Welcome to the Romana Restaurant Voice Assistant!")
#     print("You can start speaking your request in English or Hindi. Say 'exit' to end the demo.")

#     while True:
#         with microphone as source:
#             print("Listening...")
#             audio = recognizer.listen(source)

#         try:
#             query = recognizer.recognize_google(audio, language='hi-IN')
#             print(f"You said: {query}")

#             if query.lower() == "exit":
#                 print("Exiting the demo. Goodbye!")
#                 break

#             # Detect language of the query
#             detected_lang = assistant.detect_language(query)

#             # Handle the conversation with await
#             response, audio_output, conversation_history = await assistant.handle_conversation(query, [])
#             print(f"Assistant: {response}")

#             if audio_output:
#                 assistant.play_audio(audio_output)

#         except sr.UnknownValueError:
#             print("Sorry, I could not understand the audio.")
#         except sr.RequestError as e:
#             print(f"Could not request results from Google Speech Recognition service; {e}")
#         except Exception as e:
#             print(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     asyncio.run(main())