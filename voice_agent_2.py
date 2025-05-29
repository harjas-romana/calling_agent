import requests
import json
import os
import logging
import sounddevice as sd
import soundfile as sf
import io
import numpy as np
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from rag_layer_2 import TravelRAGLayer
import speech_recognition as sr

class TravelVoiceAgent:
    """
    Enhanced Voice Agent for Harjas Travels with complete booking,
    consultation, and customer service capabilities.
    """
    
    def __init__(
        self, 
        elevenlabs_api_key: str,
        rag_layer: TravelRAGLayer,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel voice by default
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
        
        # Travel agency operational data
        self.bookings = []
        self.current_booking = None
        self.consultations = []
        self.current_consultation = None
        
        # Operating hours configuration
        self.operating_hours = {
            "Monday": "9:00 AM - 6:00 PM",
            "Tuesday": "9:00 AM - 6:00 PM",
            "Wednesday": "9:00 AM - 6:00 PM",
            "Thursday": "9:00 AM - 6:00 PM",
            "Friday": "9:00 AM - 7:00 PM",
            "Saturday": "10:00 AM - 4:00 PM",
            "Sunday": "Closed"
        }
        
        self._validate_api_connection()
        self._list_audio_devices()

    def _validate_api_connection(self):
        """Validate the connection to ElevenLabs API."""
        try:
            headers = {"xi-api-key": self.elevenlabs_api_key}
            response = requests.get(f"{self.base_url}/voices", headers=headers)
            if response.status_code != 200:
                raise RuntimeError(f"API connection failed: {response.status_code} - {response.text}")
            self.logger.info("API connection validated successfully")
        except Exception as error:
            self.logger.error(f"API validation failed: {str(error)}")
            raise RuntimeError("Could not validate API connection")

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
        logger = logging.getLogger("TravelVoiceAgent")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            file_handler = logging.FileHandler(f"logs/travel_voice_agent_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler.setFormatter(formatter)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    

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
    def handle_conversation(self, query: str, conversation_history: List[Dict]) -> Tuple[str, Optional[bytes], List[Dict]]:
        """
        Main conversation handler that routes to specific functions based on context.
        Returns tuple: (response_text, audio_data, updated_history)
        """
        try:
            # Check if we're in the middle of a flight booking
            if self.current_booking and not self.current_booking.get('completed', False):
                return self._handle_booking_flow(query, conversation_history)
            
            # Check if we're in the middle of a travel consultation
            if self.current_consultation and not self.current_consultation.get('completed', False):
                return self._handle_consultation_flow(query, conversation_history)
            
            # Check for specific intents
            query_lower = query.lower()
            
            if any(word in query_lower for word in ["book", "flight", "ticket", "reservation", "travel", "trip"]):
                return self._start_booking(query, conversation_history)
            
            elif any(word in query_lower for word in ["consult", "consultation", "advice", "recommend", "suggestion"]):
                return self._start_consultation(query, conversation_history)
            
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
            
            elif any(word in query_lower for word in ["promotion", "deal", "special", "discount", "offer"]):
                return self._get_promotions(conversation_history)
            
            elif any(word in query_lower for word in ["document", "passport", "visa", "requirement"]):
                return self._get_travel_requirements(query, conversation_history)
            
            elif any(word in query_lower for word in ["cancel", "refund", "change", "reschedule"]):
                return self._handle_booking_changes(query, conversation_history)
            
            else:
                # Default to RAG response
                response, updated_history = self.rag_layer.generate_response(query, conversation_history)
                audio = self.text_to_speech(response)
                return response, audio, updated_history
                
        except Exception as e:
            self.logger.error(f"Error in conversation handling: {str(e)}")
            error_msg = "I'm sorry, I encountered an error processing your request. Could you please try again?"
            try:
                audio = self.text_to_speech(error_msg)
                return error_msg, audio, conversation_history
            except:
                return error_msg, None, conversation_history

    # Booking System
    def _start_booking(self, query: str, conversation_history: List[Dict]) -> Tuple[str, bytes, List[Dict]]:
        """Initiate flight booking process."""
        self.current_booking = {
            "completed": False,
            "step": "destination",
            "data": {}
        }
        
        response = "Thank you for choosing Harjas Travels! Let's book your trip. Where would you like to travel to? Please provide your destination city or country."
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    def _handle_booking_flow(self, query: str, conversation_history: List[Dict]) -> Tuple[str, bytes, List[Dict]]:
        """Handle multi-step booking process."""
        current_step = self.current_booking["step"]
        
        if current_step == "destination":
            # Process destination
            self.current_booking["data"]["destination"] = query
            self.current_booking["step"] = "origin"
            
            response = f"Great! You want to travel to {query}. Where will you be departing from?"
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "origin":
            # Process origin
            self.current_booking["data"]["origin"] = query
            self.current_booking["step"] = "travel_dates"
            
            response = f"You'll be traveling from {query} to {self.current_booking['data']['destination']}. When would you like to depart, and when will you return? Please provide dates like 'June 15 to June 30'."
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "travel_dates":
            try:
                # Extract dates
                dates = self._extract_dates(query)
                if len(dates) >= 2:
                    self.current_booking["data"]["departure_date"] = dates[0].strftime("%Y-%m-%d")
                    self.current_booking["data"]["return_date"] = dates[1].strftime("%Y-%m-%d")
                    self.current_booking["step"] = "num_travelers"
                    
                    response = f"You'll depart on {dates[0].strftime('%B %d, %Y')} and return on {dates[1].strftime('%B %d, %Y')}. How many travelers will be on this trip?"
                    audio = self.text_to_speech(response)
                    conversation_history.append({"role": "assistant", "content": response})
                    return response, audio, conversation_history
                else:
                    response = "I need both a departure and return date. Please specify them like 'June 15 to June 30' or 'June 15 - June 30'."
                    audio = self.text_to_speech(response)
                    return response, audio, conversation_history
            except Exception as e:
                response = "I couldn't understand those dates. Please provide them in a format like 'June 15 to June 30' or 'next Monday to Friday'."
                audio = self.text_to_speech(response)
                return response, audio, conversation_history
        
        elif current_step == "num_travelers":
            # Extract number from text
            num_travelers = self._extract_number_from_text(query)
            
            if num_travelers is not None and num_travelers > 0:
                self.current_booking["data"]["num_travelers"] = num_travelers
                self.current_booking["step"] = "traveler_names"
                self.current_booking["data"]["traveler_names"] = []
                
                response = f"We'll book for {num_travelers} travelers. Please provide the full name of traveler 1."
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
            else:
                response = "I need to know how many travelers. Please say just a number, like 'two' or 'four'."
                audio = self.text_to_speech(response)
                return response, audio, conversation_history
        
        elif current_step == "traveler_names":
            # Add traveler name
            traveler_names = self.current_booking["data"]["traveler_names"]
            traveler_names.append(query)
            
            # Check if we need more names
            if len(traveler_names) < self.current_booking["data"]["num_travelers"]:
                response = f"Thank you. Now please provide the full name of traveler {len(traveler_names) + 1}."
                audio = self.text_to_speech(response)
                return response, audio, conversation_history
            else:
                self.current_booking["step"] = "contact_info"
                response = "Thank you for providing all traveler names. What's your contact phone number?"
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
        
        elif current_step == "contact_info":
            # Simple validation - we're just checking if there are digits
            if any(char.isdigit() for char in query):
                self.current_booking["data"]["contact_phone"] = query
                self.current_booking["step"] = "email"
                
                response = "Thank you. What's your email address for booking confirmations?"
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
            else:
                response = "I need a phone number with digits. Please provide a valid phone number."
                audio = self.text_to_speech(response)
                return response, audio, conversation_history
        
        elif current_step == "email":
            # Simple email validation
            if "@" in query and "." in query:
                self.current_booking["data"]["email"] = query
                self.current_booking["step"] = "preferences"
                
                response = "Do you have any seating or meal preferences, or any special requests for your flight?"
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
            else:
                response = "That doesn't appear to be a valid email address. Please provide an email in the format: name@example.com"
                audio = self.text_to_speech(response)
                return response, audio, conversation_history
        
        elif current_step == "preferences":
            self.current_booking["data"]["preferences"] = query
            self.current_booking["step"] = "confirm"
            
            # Format confirmation message
            booking_data = self.current_booking["data"]
            traveler_list = ", ".join(booking_data["traveler_names"])
            
            response = (
                f"Let me confirm your booking details:\n"
                f"Route: {booking_data['origin']} to {booking_data['destination']}\n"
                f"Departure: {booking_data['departure_date']}\n"
                f"Return: {booking_data['return_date']}\n"
                f"Travelers: {booking_data['num_travelers']} ({traveler_list})\n"
                f"Contact: {booking_data['contact_phone']} / {booking_data['email']}\n"
                f"Preferences: {booking_data['preferences']}\n\n"
                f"Is this information correct? Please say yes or no."
            )
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "confirm":
            if "yes" in query.lower() or "correct" in query.lower() or "right" in query.lower():
                # Complete booking
                booking_id = f"HT{datetime.now().strftime('%Y%m%d%H%M%S')}"
                self.current_booking["data"]["booking_id"] = booking_id
                self.bookings.append(self.current_booking["data"])
                self.current_booking["completed"] = True
                
                response = (
                    f"Your booking is confirmed! Your booking reference number is {booking_id}. "
                    f"We'll send a confirmation email to {self.current_booking['data']['email']} shortly. "
                    f"Would you like to know about our travel insurance options or have any other questions?"
                )
                audio = self.text_to_speech(response)
                conversation_history.append({"role": "assistant", "content": response})
                return response, audio, conversation_history
            else:
                self.current_booking["step"] = "destination"
                response = "Let's start over with your booking. Where would you like to travel to?"
                audio = self.text_to_speech(response)
                return response, audio, conversation_history

    # Consultation System
    def _start_consultation(self, query: str, conversation_history: List[Dict]) -> Tuple[str, bytes, List[Dict]]:
        """Initiate travel consultation process."""
        self.current_consultation = {
            "completed": False,
            "step": "travel_interests",
            "data": {}
        }
        
        response = "I'd be happy to help you plan your perfect trip! To get started, could you tell me what type of travel experience you're interested in? For example, beach vacation, cultural tour, adventure trip, family holiday, etc."
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    def _handle_consultation_flow(self, query: str, conversation_history: List[Dict]) -> Tuple[str, bytes, List[Dict]]:
        """Handle multi-step consultation process."""
        current_step = self.current_consultation["step"]
        
        if current_step == "travel_interests":
            self.current_consultation["data"]["interests"] = query
            self.current_consultation["step"] = "destinations"
            
            response = f"Great! A {query} sounds wonderful. Do you have any specific destinations in mind, or would you like recommendations based on your interests?"
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "destinations":
            self.current_consultation["data"]["destination_preference"] = query
            self.current_consultation["step"] = "budget"
            
            response = "Thank you for sharing that. What's your approximate budget for this trip? This helps us recommend options that match your expectations."
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "budget":
            self.current_consultation["data"]["budget"] = query
            self.current_consultation["step"] = "travel_dates"
            
            response = "When are you planning to travel? Do you have specific dates in mind, or are your dates flexible?"
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "travel_dates":
            self.current_consultation["data"]["travel_dates"] = query
            self.current_consultation["step"] = "travelers"
            
            response = "How many people will be traveling? Please let me know if there are any children or seniors in your group."
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "travelers":
            self.current_consultation["data"]["travelers"] = query
            self.current_consultation["step"] = "accommodation"
            
            response = "What type of accommodation do you prefer? For example, luxury hotel, budget-friendly, resort, rental apartment, etc."
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "accommodation":
            self.current_consultation["data"]["accommodation"] = query
            self.current_consultation["step"] = "activities"
            
            response = "What activities or experiences are you most interested in during your trip? For example, sightseeing, relaxation, adventure activities, local cuisine, etc."
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "activities":
            self.current_consultation["data"]["activities"] = query
            self.current_consultation["step"] = "contact_info"
            
            response = "Thank you for sharing your preferences. To provide you with personalized recommendations, could I have your name and contact information?"
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "contact_info":
            self.current_consultation["data"]["contact_info"] = query
            self.current_consultation["step"] = "summarize"
            
            # Format consultation summary
            consult_data = self.current_consultation["data"]
            
            # Generate custom recommendations based on interests
            recommendations = self._generate_travel_recommendations(consult_data)
            
            response = (
                f"Based on your preferences:\n"
                f"- Trip type: {consult_data['interests']}\n"
                f"- Destination interest: {consult_data['destination_preference']}\n"
                f"- Budget: {consult_data['budget']}\n"
                f"- Travel dates: {consult_data['travel_dates']}\n"
                f"- Group: {consult_data['travelers']}\n"
                f"- Accommodation: {consult_data['accommodation']}\n"
                f"- Activities: {consult_data['activities']}\n\n"
                f"Here are my recommendations:\n{recommendations}\n\n"
                f"Would you like me to email these recommendations to you or would you prefer to speak with one of our travel consultants for more detailed planning?"
            )
            
            self.current_consultation["data"]["recommendations"] = recommendations
            
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history
        
        elif current_step == "summarize":
            self.current_consultation["data"]["follow_up_preference"] = query
            self.current_consultation["completed"] = True
            
            # Store the consultation for future reference
            self.consultations.append(self.current_consultation["data"])
            
            if "email" in query.lower() or "send" in query.lower():
                response = (
                    "Perfect! I'll arrange for these recommendations to be emailed to you shortly. "
                    "Is there anything specific you'd like our travel consultant to focus on when they review your preferences? "
                    "They'll reach out within 24 hours to discuss your trip further."
                )
            else:
                response = (
                    "I'll connect you with one of our expert travel consultants who specializes in your type of trip. "
                    "They'll contact you within 24 hours to discuss your preferences in more detail and help craft your perfect itinerary. "
                    "Is there a preferred time for them to call you?"
                )
            
            audio = self.text_to_speech(response)
            conversation_history.append({"role": "assistant", "content": response})
            return response, audio, conversation_history

    # Feedback System
    def _handle_feedback(self, query: str, conversation_history: List[Dict]) -> Tuple[str, bytes, List[Dict]]:
        """Handle customer feedback collection."""
        response = (
            "Thank you for taking the time to share your feedback with Harjas Travels. "
            "Your input helps us improve our services. Could you tell me more about your experience with us? "
            "What did you enjoy most, and is there anything we could improve?"
        )
        
        # In a real implementation, you would store this feedback
        self.logger.info(f"Customer feedback received: {query}")
        
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    # Handle booking changes/cancellations
    def _handle_booking_changes(self, query: str, conversation_history: List[Dict]) -> Tuple[str, bytes, List[Dict]]:
        """Handle booking changes, cancellations, or refunds."""
        response = (
            "I understand you'd like to make changes to your booking. To proceed, I'll need your booking reference number. "
            "Alternatively, I can look up your booking using your full name and travel dates. "
            "Please note that changes and cancellations are subject to our policies and may incur fees depending on your fare type and the airline's rules."
        )
        
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    # Travel requirements information
    def _get_travel_requirements(self, query: str, conversation_history: List[Dict]) -> Tuple[str, bytes, List[Dict]]:
        """Provide information about travel documentation requirements."""
        # Extract country name from query if present
        countries = self.rag_layer.knowledge_base["agency_info"]["popular_countries"]
        mentioned_country = None
        
        for country in countries:
            if country.lower() in query.lower():
                mentioned_country = country
                break
        
        if mentioned_country:
            response = (
                f"For travel to {mentioned_country}, you'll typically need:\n"
                f"1. A passport valid for at least 6 months beyond your stay\n"
                f"2. Visa requirements vary based on your citizenship\n"
                f"3. Return or onward tickets\n"
                f"4. Proof of sufficient funds\n\n"
                f"For the most up-to-date and specific requirements based on your citizenship, I recommend checking the official government website or consulate of {mentioned_country}. "
                f"Would you like me to check specific visa requirements based on your nationality?"
            )
        else:
            response = (
                "For international travel, you'll typically need:\n"
                "1. A valid passport (usually valid for at least 6 months beyond your stay)\n"
                "2. Visas or travel authorizations (requirements vary by destination and your citizenship)\n"
                "3. Return or onward travel tickets\n"
                "4. Travel insurance (highly recommended and sometimes mandatory)\n"
                "5. Vaccination certificates (for certain destinations)\n\n"
                "Which country are you planning to visit? I can provide more specific information based on your destination."
            )
        
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    # Promotions and deals
    def _get_promotions(self, conversation_history: List[Dict]) -> Tuple[str, bytes, List[Dict]]:
        """Provide information about current promotions and deals."""
        promotions = self.rag_layer.knowledge_base["promotions"]
        
        if promotions:
            promo_text = "\n".join([f"- {promo['name']}: {promo['details']}" for promo in promotions])
            response = (
                f"Here are our current promotions at Harjas Travels:\n{promo_text}\n\n"
                f"We also have exclusive deals with certain airlines and hotels that aren't advertised. "
                f"Would you like to hear about destination-specific offers or would you like to book a trip taking advantage of these promotions?"
            )
        else:
            response = (
                "While we don't have any public promotions at the moment, we do offer personalized deals based on your travel preferences. "
                "Our partnerships with airlines and hotels allow us to create custom packages with special pricing. "
                "Where are you interested in traveling to? I'd be happy to check for any unadvertised deals for that destination."
            )
        
        audio = self.text_to_speech(response)
        conversation_history.append({"role": "assistant", "content": response})
        return response, audio, conversation_history

    # Available commands
    def _get_available_commands(self) -> str:
        """Return list of available commands and features."""
        return (
            "Here are some things you can ask me about Harjas Travels:\n"
            "- 'Book a flight' or 'I need to travel to [destination]'\n"
            "- 'I need travel advice' or 'Help me plan a trip'\n"
            "- 'What promotions do you have?'\n"
            "- 'What are your business hours?'\n"
            "- 'What documents do I need for international travel?'\n"
            "- 'I need to change/cancel my booking'\n"
            "- 'Do you offer travel insurance?'\n"
            "- 'I have feedback about your service'\n"
            "- Ask any questions about destinations, travel requirements, or our services!"
        )

    # Get operating hours
    def _get_operating_hours(self) -> str:
        """Return formatted operating hours."""
        hours_text = "\n".join(f"{day}: {hours}" for day, hours in self.operating_hours.items())
        return f"Harjas Travels operating hours are:\n{hours_text}\n\nHow can I assist you with your travel needs today?"

    # Generate travel recommendations
    def _generate_travel_recommendations(self, preferences: Dict) -> str:
        """Generate customized travel recommendations based on user preferences."""
        interests = preferences.get('interests', '').lower()
        destination_preference = preferences.get('destination_preference', '').lower()
        budget = preferences.get('budget', '').lower()
        activities = preferences.get('activities', '').lower()
        
        # Default recommendations if no specific preferences are given
        recommendations = []
        
        # Check for beach/relaxation interests
        if any(word in interests for word in ['beach', 'relax', 'resort', 'tropical']):
            if 'low' in budget or 'budget' in budget:
                recommendations.append("- Phuket, Thailand: Affordable beach resorts with excellent value")
                recommendations.append("- Goa, India: Beautiful beaches with budget-friendly accommodations")
            elif 'high' in budget or 'luxury' in budget:
                recommendations.append("- Maldives: Exclusive private island resorts with overwater bungalows")
                recommendations.append("- Santorini, Greece: Luxury cliffside accommodations with stunning views")
            else:  # Medium budget
                recommendations.append("- Bali, Indonesia: Beautiful beaches with a range of accommodation options")
                recommendations.append("- Cancun, Mexico: All-inclusive resorts with pristine Caribbean beaches")
        
        # Check for cultural/historical interests
        if any(word in interests for word in ['culture', 'history', 'museum', 'historical']):
            if 'low' in budget or 'budget' in budget:
                recommendations.append("- Hanoi, Vietnam: Rich culture and history with affordable accommodations")
                recommendations.append("- Krakow, Poland: Preserved medieval architecture and museums at reasonable prices")
            elif 'high' in budget or 'luxury' in budget:
                recommendations.append("- Kyoto, Japan: Traditional ryokans and cultural experiences with luxury service")
                recommendations.append("- Rome, Italy: Five-star hotels near ancient ruins and world-class museums")
            else:  # Medium budget
                recommendations.append("- Istanbul, Turkey: Where East meets West with stunning historical sites")
                recommendations.append("- Prague, Czech Republic: Well-preserved historical center with reasonable prices")
        
        # Check for adventure interests
        if any(word in interests for word in ['adventure', 'hiking', 'trek', 'outdoor']):
            if 'low' in budget or 'budget' in budget:
                recommendations.append("- Nepal: World-class trekking with affordable teahouse accommodations")
                recommendations.append("- Colombia: Emerging adventure destination with competitive prices")
            elif 'high' in budget or 'luxury' in budget:
                recommendations.append("- New Zealand: Luxury lodges with private adventure experiences")
                recommendations.append("- Costa Rica: Eco-luxury resorts with private rainforest and wildlife tours")
            else:  # Medium budget
                recommendations.append("- Peru: Machu Picchu treks with comfortable accommodations")
                recommendations.append("- South Africa: Safari experiences with mid-range lodging options")
        
        # Check for family-friendly interests
        if any(word in interests for word in ['family', 'kid', 'children']):
            if 'low' in budget or 'budget' in budget:
                recommendations.append("- Orlando, FL: Theme parks with affordable off-site accommodations")
                recommendations.append("- Phuket, Thailand: Family-friendly resorts with excellent value")
            elif 'high' in budget or 'luxury' in budget:
                recommendations.append("- Maldives: Family-friendly luxury resorts with kids clubs and activities")
                recommendations.append("- Switzerland: Luxury family accommodations with outdoor activities")
            else:  # Medium budget
                recommendations.append("- Barcelona, Spain: Culture, beaches and family attractions")
                recommendations.append("- Gold Coast, Australia: Theme parks and beaches for all ages")
        
        # If specific destination was mentioned, prioritize it
        if destination_preference and not all(word in destination_preference for word in ['not sure', 'recommend', 'don\'t know']):
            for country in self.rag_layer.knowledge_base["agency_info"]["popular_countries"]:
                if country.lower() in destination_preference.lower():
                    recommendations = [rec for rec in recommendations if country in rec] or recommendations
                    recommendations.insert(0, f"Since you mentioned {country}, we highly recommend exploring options there based on your preferences.")
                    break
        
        # If no matches found, provide general recommendations
        if not recommendations:
            recommendations = [
                "- Paris, France: The perfect blend of culture, cuisine, and iconic sights",
                "- Barcelona, Spain: Beautiful architecture, beaches, and vibrant culture",
                "- Tokyo, Japan: Fascinating blend of traditional and ultra-modern experiences",
                "- New York City, USA: World-class attractions, dining, and entertainment"
            ]
        
        # Add activity-specific recommendations
        if 'food' in activities or 'cuisine' in activities or 'dining' in activities:
            recommendations.append("For food lovers: Consider a food tour or cooking class in your destination to experience authentic local cuisine.")
        
        if 'sightseeing' in activities:
            recommendations.append("For sightseeing: We recommend booking skip-the-line tickets for major attractions to maximize your time.")
        
        # Add package information
        recommendations.append("\nWe can arrange complete packages including flights, accommodations, transfers, and activities tailored to your preferences.")
        
        return "\n".join(recommendations)

    # Helper methods for parsing user input
    def _extract_dates(self, date_text: str) -> List[datetime]:
        """Extract dates from text input like 'June 15 to June 30'."""
        dates = []
        current_year = datetime.now().year
        
        # Common date format patterns
        patterns = [
            r'(\w+ \d{1,2})(?:st|nd|rd|th)? (?:to|through|-)? ?(\w+ \d{1,2})(?:st|nd|rd|th)?',  # June 15 to June 30
            r'(\d{1,2})(?:st|nd|rd|th)? (?:of )?\w+ (?:to|through|-)? ?(\d{1,2})(?:st|nd|rd|th)? (?:of )?\w+',  # 15th June to 30th June
            r'(\d{1,2}/\d{1,2})(?:/\d{2,4})? (?:to|through|-)? ?(\d{1,2}/\d{1,2})(?:/\d{2,4})?'  # 6/15 to 6/30
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_text)
            if match:
                try:
                    start_date_str = match.group(1)
                    end_date_str = match.group(2)
                    
                    # Parse dates - this is simplified and would need more robust parsing in production
                    start_date = datetime.strptime(f"{start_date_str} {current_year}", "%B %d %Y")
                    end_date = datetime.strptime(f"{end_date_str} {current_year}", "%B %d %Y")
                    
                    # Handle year wrap-around (December to January)
                    if end_date < start_date:
                        end_date = end_date.replace(year=current_year + 1)
                    
                    dates = [start_date, end_date]
                    break
                except ValueError:
                    continue
        
        # Handle relative dates like "next week" or "this weekend"
        if not dates:
            today = datetime.now()
            
            if "weekend" in date_text.lower():
                # Find next Saturday
                days_until_saturday = (5 - today.weekday()) % 7
                if days_until_saturday == 0:
                    days_until_saturday = 7
                
                start_date = today + timedelta(days=days_until_saturday)
                end_date = start_date + timedelta(days=2)  # Until Monday
                dates = [start_date, end_date]
            
            elif "next week" in date_text.lower():
                # Start next Monday
                days_until_monday = (0 - today.weekday()) % 7
                if days_until_monday == 0:
                    days_until_monday = 7
                
                start_date = today + timedelta(days=days_until_monday)
                end_date = start_date + timedelta(days=6)  # Until Sunday
                dates = [start_date, end_date]
        
        return dates
    
    def _extract_number_from_text(self, text: str) -> Optional[int]:
        """Extract a number from text, handling both digits and word numbers."""
        # Check for digits
        digit_match = re.search(r'\b(\d+)\b', text)
        if digit_match:
            return int(digit_match.group(1))
        
        # Check for word numbers
        word_to_number = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        for word, number in word_to_number.items():
            if word in text.lower():
                return number
        
        return None

    # Voice input handling
    def listen_for_voice_input(self) -> str:
        """Listen for voice input and convert to text."""
        recognizer = sr.Recognizer()
        
        try:
            with sr.Microphone() as source:
                print("Listening...")
                self.logger.info("Listening for voice input")
                
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
                
                print("Processing speech...")
                self.logger.info("Processing speech to text")
                
                # Convert speech to text
                text = recognizer.recognize_google(audio)
                
                print(f"You said: {text}")
                self.logger.info(f"Voice input received: {text}")
                
                return text
        
        except sr.WaitTimeoutError:
            message = "Sorry, I didn't hear anything. Could you please speak again?"
            print(message)
            self.logger.warning("No voice input detected (timeout)")
            return ""
            
        except sr.UnknownValueError:
            message = "Sorry, I couldn't understand what you said. Could you please repeat that?"
            print(message)
            self.logger.warning("Voice input not recognized")
            return ""
            
        except sr.RequestError as e:
            message = f"Sorry, there was an error with the speech recognition service: {str(e)}"
            print(message)
            self.logger.error(f"Speech recognition service error: {str(e)}")
            return ""
            
        except Exception as e:
            message = "Sorry, there was an unexpected error processing your voice input."
            print(message)
            self.logger.error(f"Voice input processing error: {str(e)}")
            return ""

    # Main voice interaction loop
    def start_voice_interaction(self) -> None:
        """Start voice interaction loop."""
        conversation_history = []
        
        # Initial greeting
        greeting = "Hello! Welcome to Harjas Travels. I'm your virtual travel assistant. How may I help you today?"
        print("Assistant: " + greeting)
        
        audio = self.text_to_speech(greeting)
        self.play_audio(audio)
        
        conversation_history.append({"role": "assistant", "content": greeting})
        
        try:
            while True:
                # Listen for user input
                user_input = self.listen_for_voice_input()
                
                if not user_input:
                    continue
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Check for exit command
                if any(word in user_input.lower() for word in ["exit", "quit", "goodbye", "bye"]):
                    farewell = "Thank you for contacting Harjas Travels. Have a wonderful day!"
                    print("Assistant: " + farewell)
                    
                    audio = self.text_to_speech(farewell)
                    self.play_audio(audio)
                    
                    break
                
                # Process query and get response
                response, audio, conversation_history = self.handle_conversation(
                    user_input, conversation_history
                )
                
                print("Assistant: " + response)
                
                # Play audio response
                if audio:
                    self.play_audio(audio)
                    
        except KeyboardInterrupt:
            print("\nVoice interaction ended by user.")
        except Exception as e:
            self.logger.error(f"Error in voice interaction loop: {str(e)}")
            print(f"Error: {str(e)}")
            print("Voice interaction ended due to an error.")

# Example usage of the TravelVoiceAgent
if __name__ == "__main__":
    try:
        # Get API key from environment variable
        elevenlabs_api_key = "sk_a643471cf3d2de658ac47648b33d8314bfe39dcc14ebfe7b"

        # Initialize the RAG layer
        rag_layer = TravelRAGLayer(elevenlabs_api_key)
        
        
        
        if not elevenlabs_api_key:
            print("Warning: ELEVENLABS_API_KEY environment variable not set.")
            print("Please set the environment variable or provide the API key directly.")
            elevenlabs_api_key = input("Enter your ElevenLabs API key: ")
            
        # Create travel voice agent
        agent = TravelVoiceAgent(
            elevenlabs_api_key=elevenlabs_api_key,
            rag_layer=rag_layer,
            voice_id="21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        )
        
        print("-" * 50)
        print("Harjas Travels Voice Agent Initialized")
        print("Say 'exit', 'quit', or 'goodbye' to end the session")
        print("-" * 50)
        
        # Start voice interaction loop
        agent.start_voice_interaction()
        
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"Error initializing TravelVoiceAgent: {str(e)}")
        import traceback
        traceback.print_exc()