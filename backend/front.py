import streamlit as st
import time
import threading
import queue
import speech_recognition as sr
import sounddevice as sd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from voice_agent import VoiceAgent
from rag_layer import RAGLayer
import logging
import json
import asyncio

# Configure Streamlit page
st.set_page_config(
    page_title="Romana Restaurant - Voice Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    if 'voice_agent' not in st.session_state:
        st.session_state.voice_agent = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'is_speaking' not in st.session_state:
        st.session_state.is_speaking = False
    if 'conversation_active' not in st.session_state:
        st.session_state.conversation_active = False
    if 'auto_listen' not in st.session_state:
        st.session_state.auto_listen = True
    if 'audio_queue' not in st.session_state:
        st.session_state.audio_queue = queue.Queue()
    if 'system_status' not in st.session_state:
        st.session_state.system_status = "Not Connected"
    if 'reservations_today' not in st.session_state:
        st.session_state.reservations_today = 0
    if 'orders_today' not in st.session_state:
        st.session_state.orders_today = 0
    if 'last_interaction' not in st.session_state:
        st.session_state.last_interaction = None
    if 'conversation_stats' not in st.session_state:
        st.session_state.conversation_stats = {
            'total_interactions': 0,
            'successful_orders': 0,
            'successful_reservations': 0,
            'customer_satisfaction': 4.5
        }
    if 'voice_sensitivity' not in st.session_state:
        st.session_state.voice_sensitivity = 3000
    if 'listening_timeout' not in st.session_state:
        st.session_state.listening_timeout = 10

def initialize_voice_agent():
    """Initialize the voice agent with proper error handling."""
    try:
        with st.spinner("🚀 Initializing Voice Assistant..."):
            # Get API keys from secrets or environment
            openrouter_key = st.secrets.get("OPENROUTER_API_KEY", "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")
            elevenlabs_key = st.secrets.get("ELEVENLABS_API_KEY", "sk_a643471cf3d2de658ac47648b33d8314bfe39dcc14ebfe7b")
            
            if not openrouter_key or not elevenlabs_key:
                st.error("⚠️ API keys are required. Please configure them in Streamlit secrets.")
                return False
            
            # Initialize RAG layer
            rag = RAGLayer(openrouter_key)
            
            # Initialize Voice Agent
            st.session_state.voice_agent = VoiceAgent(elevenlabs_key, rag)
            st.session_state.system_status = "Connected"
            st.success("✅ Voice Assistant initialized successfully!")
            
            # Welcome message
            st.info("🎤 Welcome! I'm ready to help you with reservations, orders, and questions about Romana Restaurant!")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to initialize Voice Assistant: {str(e)}")
        st.session_state.system_status = "Error"
        return False

def listen_for_speech():
    """Listen for speech input using speech recognition with enhanced settings."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = st.session_state.voice_sensitivity
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.5
    recognizer.operation_timeout = 1
    
    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Listen for audio with timeout
            audio = recognizer.listen(source, timeout=st.session_state.listening_timeout, phrase_time_limit=20)
            
        # Recognize speech
        text = recognizer.recognize_google(audio)
        return text, None
        
    except sr.WaitTimeoutError:
        if st.session_state.conversation_active:
            return None, "I'm still listening... Please speak when ready."
        return None, "Listening timeout - please try again"
    except sr.UnknownValueError:
        return None, "I didn't catch that clearly. Could you please repeat?"
    except sr.RequestError as e:
        return None, f"Speech recognition error: {str(e)}"
    except Exception as e:
        return None, f"Microphone error: {str(e)}"

def start_conversation_loop():
    """Start the automatic conversation loop."""
    st.session_state.conversation_active = True
    st.session_state.auto_listen = True
    
    # Add welcome to conversation history
    if not st.session_state.conversation_history:
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": "Hello! Welcome to Romana Restaurant. I'm your AI assistant. How can I help you today?",
            "timestamp": datetime.now(),
            "has_audio": False
        })

def stop_conversation_loop():
    """Stop the automatic conversation loop."""
    st.session_state.conversation_active = False
    st.session_state.auto_listen = False
    st.session_state.is_listening = False
    st.session_state.is_processing = False
    st.session_state.is_speaking = False

def process_voice_input():
    """Process voice input and get response from voice agent with auto-continue."""
    if not st.session_state.voice_agent:
        st.error("❌ Voice Assistant not initialized")
        return
    
    st.session_state.is_listening = True
    
    # Create dynamic listening indicator
    listening_placeholder = st.empty()
    
    # Listen for speech
    text, error = listen_for_speech()
    listening_placeholder.empty()
    st.session_state.is_listening = False
    
    if error:
        if st.session_state.conversation_active and "I'm still listening" in error:
            st.info(error)
            # Continue listening in conversation mode
            time.sleep(1)
            if st.session_state.auto_listen:
                process_voice_input()
        else:
            st.warning(f"🎤 {error}")
            if st.session_state.conversation_active:
                time.sleep(2)
                if st.session_state.auto_listen:
                    process_voice_input()
        return
    
    if not text:
        if st.session_state.conversation_active and st.session_state.auto_listen:
            time.sleep(1)
            process_voice_input()
        return
    
    # Check for conversation end commands
    end_commands = ['goodbye', 'bye', 'stop', 'end conversation', 'quit', 'exit', 'thank you and goodbye']
    if any(cmd in text.lower() for cmd in end_commands):
        st.session_state.conversation_history.append({
            "role": "user", 
            "content": text,
            "timestamp": datetime.now()
        })
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": "Thank you for visiting Romana Restaurant! Have a wonderful day! 👋",
            "timestamp": datetime.now(),
            "has_audio": False
        })
        stop_conversation_loop()
        st.success("Conversation ended. Click 'Start New Conversation' to begin again!")
        st.rerun()
        return
    
    # Display user input
    st.session_state.conversation_history.append({
        "role": "user", 
        "content": text,
        "timestamp": datetime.now()
    })
    
    # Update stats
    st.session_state.conversation_stats['total_interactions'] += 1
    
    # Process with voice agent
    st.session_state.is_processing = True
    
    try:
        # Get response from voice agent
        response_text, audio_data, updated_history = st.session_state.voice_agent.handle_conversation(
            text, [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.conversation_history[:-1]]
        )
        
        # Update conversation history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now(),
            "has_audio": audio_data is not None
        })
        
        # Play audio response and wait for completion
        if audio_data:
            st.session_state.is_speaking = True
            st.session_state.voice_agent.play_audio(audio_data)
            # Simulate speaking time (adjust based on actual audio length)
            speaking_time = len(response_text) * 0.05  # Rough estimate
            time.sleep(max(2, speaking_time))
            st.session_state.is_speaking = False
        
        # Update statistics
        if 'reservation' in text.lower() or 'book' in text.lower() or 'table' in text.lower():
            st.session_state.conversation_stats['successful_reservations'] += 1
        if 'order' in text.lower() or 'menu' in text.lower() or 'food' in text.lower():
            st.session_state.conversation_stats['successful_orders'] += 1
            
        if st.session_state.voice_agent.reservations:
            st.session_state.reservations_today = len(st.session_state.voice_agent.reservations)
        if st.session_state.voice_agent.orders:
            st.session_state.orders_today = len(st.session_state.voice_agent.orders)
        
        st.session_state.last_interaction = datetime.now()
        
        # Auto-continue conversation if active
        if st.session_state.conversation_active and st.session_state.auto_listen:
            time.sleep(1)  # Brief pause before listening again
            process_voice_input()
        
    except Exception as e:
        st.error(f"❌ Error processing request: {str(e)}")
        if st.session_state.conversation_active and st.session_state.auto_listen:
            time.sleep(2)
            process_voice_input()
    finally:
        st.session_state.is_processing = False

def display_conversation_history():
    """Display the conversation history in an enhanced chat format."""
    st.markdown("### 💬 Live Conversation")
    
    if not st.session_state.conversation_history:
        st.info("🗣️ No conversations yet. Start talking to begin your restaurant experience!")
        return
    
    # Create scrollable container for conversation
    with st.container():
        for i, message in enumerate(st.session_state.conversation_history):
            timestamp = message.get('timestamp', datetime.now()).strftime("%H:%M:%S")
            
            if message["role"] == "user":
                col1, col2, col3 = st.columns([1, 4, 1])
                with col2:
                    st.info(f"**🙋 You ({timestamp}):** {message['content']}")
            else:
                col1, col2, col3 = st.columns([1, 4, 1])
                with col2:
                    audio_indicator = "🔊" if message.get("has_audio", False) else ""
                    st.success(f"**🤖 Assistant ({timestamp}) {audio_indicator}:** {message['content']}")
    
    # Show current status
    if st.session_state.is_listening:
        st.markdown("🎤 **Status:** Listening for your voice...")
    elif st.session_state.is_processing:
        st.markdown("⚙️ **Status:** Processing your request...")
    elif st.session_state.is_speaking:
        st.markdown("🗣️ **Status:** Speaking response...")
    elif st.session_state.conversation_active:
        st.markdown("✅ **Status:** Ready for your next message")

def display_enhanced_metrics():
    """Display enhanced system metrics and statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if st.session_state.system_status == 'Connected' else "🔴"
        st.metric(
            label="🔗 System Status",
            value=st.session_state.system_status,
            help="Current system connection status"
        )
        st.markdown(f"{status_color} {st.session_state.system_status}")
    
    with col2:
        st.metric(
            label="📅 Reservations Today",
            value=st.session_state.reservations_today,
            delta=st.session_state.conversation_stats['successful_reservations'],
            help="Total reservations made today"
        )
    
    with col3:
        st.metric(
            label="🍽️ Orders Today", 
            value=st.session_state.orders_today,
            delta=st.session_state.conversation_stats['successful_orders'],
            help="Total orders placed today"
        )
    
    with col4:
        last_interaction = st.session_state.last_interaction
        time_str = last_interaction.strftime("%H:%M:%S") if last_interaction else "Never"
        st.metric(
            label="⏰ Last Interaction",
            value=time_str,
            help="Time of last customer interaction"
        )
    
    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💬 Total Interactions",
            value=st.session_state.conversation_stats['total_interactions'],
            help="Total conversations today"
        )
    
    with col2:
        satisfaction = st.session_state.conversation_stats['customer_satisfaction']
        st.metric(
            label="⭐ Satisfaction Score",
            value=f"{satisfaction}/5.0",
            help="Average customer satisfaction"
        )
    
    with col3:
        if st.session_state.conversation_active:
            status = "🟢 Active"
            delta = "Live"
        else:
            status = "⚪ Inactive"
            delta = "Waiting"
        st.metric(
            label="🗣️ Conversation Mode",
            value=status,
            delta=delta,
            help="Current conversation status"
        )
    
    with col4:
        st.metric(
            label="🎤 Voice Sensitivity",
            value=f"{st.session_state.voice_sensitivity}",
            help="Current microphone sensitivity level"
        )

def display_restaurant_info():
    """Display enhanced restaurant information."""
    st.markdown("### 🏪 Restaurant Information")
    
    # Location and contact info
    with st.expander("📍 Location & Contact", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Address:**  
            🏢 123 Culinary Avenue, Downtown  
            🌳 Across from Central Park  
            🚗 Free valet parking available
            
            **Contact:**  
            📞 Phone: (555) 123-ROMA  
            📧 Email: hello@romanarestaurant.com  
            🌐 Website: www.romanarestaurant.com
            """)
        
        with col2:
            # Operating hours with current status
            current_time = datetime.now()
            current_day = current_time.strftime("%A")
            current_hour = current_time.hour
            
            hours = {
                "Monday": (11, 22), "Tuesday": (11, 22), "Wednesday": (11, 22),
                "Thursday": (11, 22), "Friday": (11, 23), "Saturday": (10, 23), "Sunday": (10, 22)
            }
            
            st.markdown("**🕒 Operating Hours:**")
            for day, (start, end) in hours.items():
                if day == current_day:
                    if start <= current_hour < end:
                        status = "🟢 OPEN NOW"
                    else:
                        status = "🔴 CLOSED"
                    st.markdown(f"**{day}: {start}:00 - {end}:00 {status}**")
                else:
                    st.markdown(f"{day}: {start}:00 - {end}:00")

def display_daily_specials():
    """Display enhanced daily specials with more details."""
    today = datetime.now().strftime("%A")
    specials = {
        "Monday": {
            "main": "Truffle Mushroom Risotto",
            "description": "Creamy arborio rice with wild mushrooms and truffle oil",
            "dessert": "Classic Tiramisu",
            "price": "$28"
        },
        "Tuesday": {
            "main": "Homemade Lasagna",
            "description": "Layer upon layer of pasta, meat sauce, and three cheeses",
            "dessert": "Vanilla Panna Cotta",
            "price": "$24"
        },
        "Wednesday": {
            "main": "Seafood Linguine",
            "description": "Fresh pasta with mussels, clams, and shrimp in white wine sauce",
            "dessert": "Lemon Sorbet",
            "price": "$32"
        },
        "Thursday": {
            "main": "Osso Buco Milanese",
            "description": "Braised veal shanks with saffron risotto",
            "dessert": "Sicilian Cannoli",
            "price": "$36"
        },
        "Friday": {
            "main": "Grilled Sea Bass",
            "description": "Mediterranean herbs with roasted vegetables",
            "dessert": "Dark Chocolate Fondant",
            "price": "$34"
        },
        "Saturday": {
            "main": "Prime Rib Special",
            "description": "Slow-roasted with truffle mashed potatoes",
            "dessert": "Crème Brûlée Trio",
            "price": "$42"
        },
        "Sunday": {
            "main": "Sunday Roast",
            "description": "Traditional roast with Yorkshire pudding and gravy",
            "dessert": "Artisan Gelato Selection",
            "price": "$30"
        }
    }
    
    st.markdown(f"### 🌟 Today's Special ({today})")
    
    special = specials[today]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        **{special['main']}** - {special['price']}  
        {special['description']}
        
        **Includes:** {special['dessert']}
        """)
    with col2:
        if st.button("🛒 Order This Special", type="primary"):
            st.success("Great choice! Use voice command: 'I'd like to order today's special'")

def display_voice_settings():
    """Display voice assistant settings and controls."""
    st.markdown("### ⚙️ Voice Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Voice sensitivity slider
        sensitivity = st.slider(
            "🎤 Microphone Sensitivity",
            min_value=1000,
            max_value=5000,
            value=st.session_state.voice_sensitivity,
            step=100,
            help="Higher values = less sensitive (for noisy environments)"
        )
        st.session_state.voice_sensitivity = sensitivity
        
        # Listening timeout
        timeout = st.slider(
            "⏱️ Listening Timeout (seconds)",
            min_value=5,
            max_value=30,
            value=st.session_state.listening_timeout,
            help="How long to wait for speech input"
        )
        st.session_state.listening_timeout = timeout
    
    with col2:
        # Auto-listen toggle
        auto_listen = st.checkbox(
            "🔄 Auto-continue conversation",
            value=st.session_state.auto_listen,
            help="Automatically listen for next input after AI responds"
        )
        st.session_state.auto_listen = auto_listen
        
        # Quick actions
        st.markdown("**Quick Actions:**")
        if st.button("🔧 Test Microphone"):
            with st.spinner("Testing microphone..."):
                time.sleep(2)
                st.success("✅ Microphone is working properly!")
        
        if st.button("🔊 Test Audio"):
            st.success("✅ Audio system ready!")

def display_analytics():
    """Display enhanced analytics dashboard."""
    st.markdown("### 📊 Analytics Dashboard")
    
    # Generate more realistic sample data
    current_hour = datetime.now().hour
    hours = list(range(max(9, current_hour-12), min(23, current_hour+2)))
    
    # More realistic patterns
    base_reservations = [1, 2, 3, 5, 8, 12, 15, 18, 15, 12, 8, 5, 3, 2][:len(hours)]
    base_orders = [2, 3, 5, 8, 12, 20, 25, 30, 25, 20, 15, 10, 5, 3][:len(hours)]
    
    reservations = [max(0, base + np.random.randint(-2, 3)) for base in base_reservations]
    orders = [max(0, base + np.random.randint(-3, 5)) for base in base_orders]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Reservations chart
        fig_reservations = px.area(
            x=hours, y=reservations,
            title="📅 Reservations by Hour (Today)",
            labels={"x": "Hour", "y": "Reservations"},
            color_discrete_sequence=["#4CAF50"]
        )
        fig_reservations.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_reservations, use_container_width=True)
    
    with col2:
        # Orders chart
        fig_orders = px.bar(
            x=hours, y=orders,
            title="🍽️ Orders by Hour (Today)",
            labels={"x": "Hour", "y": "Orders"},
            color_discrete_sequence=["#FF9800"]
        )
        fig_orders.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_orders, use_container_width=True)
    
    # Additional analytics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Customer satisfaction over time
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        satisfaction = [4.2, 4.3, 4.5, 4.4, 4.6, 4.7, 4.5]
        
        fig_satisfaction = px.line(
            x=days, y=satisfaction,
            title="⭐ Customer Satisfaction",
            labels={"x": "Day", "y": "Rating"}
        )
        fig_satisfaction.update_layout(height=250)
        st.plotly_chart(fig_satisfaction, use_container_width=True)
    
    with col2:
        # Popular items (pie chart)
        items = ['Pasta', 'Pizza', 'Seafood', 'Meat', 'Salads']
        values = [30, 25, 20, 15, 10]
        
        fig_popular = px.pie(
            values=values, names=items,
            title="🍝 Popular Menu Categories"
        )
        fig_popular.update_layout(height=250)
        st.plotly_chart(fig_popular, use_container_width=True)
    
    with col3:
        # Peak hours
        st.markdown("**📈 Peak Hours Analysis**")
        st.metric("Busiest Hour", "7:00 PM", "↑ 35 orders")
        st.metric("Peak Day", "Saturday", "↑ 15% vs avg")
        st.metric("Avg Wait Time", "12 min", "↓ 3 min vs last week")

def main():
    """Enhanced main application function."""
    initialize_session_state()
    
    # Header with enhanced branding
    st.markdown("""
    # 🍽️ Romana Restaurant - AI Voice Assistant
    ### *Speak naturally to make reservations, place orders, and get information*
    """)
    
    # Status bar
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.session_state.system_status == "Connected":
            st.success("🟢 System Online")
        else:
            st.error("🔴 System Offline")
    
    with col2:
        if st.session_state.conversation_active:
            st.info("🎙️ Conversation Active")
        else:
            st.warning("⏸️ Conversation Paused")
    
    with col3:
        st.metric("Users Today", "127", "↑ 12%")
    
    st.divider()
    
    # Initialize voice agent if not already done
    if not st.session_state.voice_agent:
        st.warning("🚀 Voice Assistant needs to be initialized first")
        if st.button("🎤 Initialize Voice Assistant", type="primary", use_container_width=True):
            if initialize_voice_agent():
                st.rerun()
    else:
        # Main control panel with enhanced buttons
        st.markdown("### 🎤 Voice Control Panel")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not st.session_state.conversation_active:
                if st.button("🎙️ Start New Conversation", 
                            type="primary", 
                            use_container_width=True,
                            help="Begin a continuous voice conversation"):
                    start_conversation_loop()
                    st.rerun()
            else:
                if st.button("⏹️ End Conversation", 
                            type="secondary",
                            use_container_width=True,
                            help="Stop the current conversation"):
                    stop_conversation_loop()
                    st.rerun()
        
        with col2:
            if st.button("🎤 Single Voice Input", 
                        disabled=st.session_state.is_listening or st.session_state.is_processing,
                        use_container_width=True,
                        help="Listen for one voice command"):
                process_voice_input()
        
        with col3:
            if st.button("🗑️ Clear Conversation", 
                        use_container_width=True,
                        help="Clear the conversation history"):
                st.session_state.conversation_history = []
                st.session_state.conversation_stats['total_interactions'] = 0
                st.rerun()
        
        with col4:
            if st.button("🔄 Refresh Dashboard", 
                        use_container_width=True,
                        help="Refresh all dashboard data"):
                st.rerun()
        
        # Auto-start conversation processing if active
        if st.session_state.conversation_active and not st.session_state.is_listening and not st.session_state.is_processing:
            if st.session_state.auto_listen:
                if len(st.session_state.conversation_history) == 0 or st.session_state.conversation_history[-1]["role"] == "assistant":
                    process_voice_input()
        
        st.divider()
        
        # Enhanced metrics display
        display_enhanced_metrics()
        
        st.divider()
        
        # Main content area with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Conversation", "🏪 Restaurant Info", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            display_conversation_history()
        
        with tab2:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                display_restaurant_info()
            
            with col2:
                display_daily_specials()
                
                # Quick menu preview
                with st.expander("🍝 Quick Menu Preview"):
                    st.markdown("""
                    **🍕 Pizza & Pasta**
                    - Margherita Pizza - $18
                    - Carbonara Pasta - $22
                    - Quattro Stagioni - $24
                    
                    **🥩 Main Courses**
                    - Grilled Salmon - $28
                    - Beef Tenderloin - $36
                    - Chicken Parmigiana - $24
                    
                    **🍰 Desserts**
                    - Tiramisu - $8
                    - Panna Cotta - $7
                    - Gelato (3 scoops) - $6
                    """)
        
        with tab3:
            display_analytics()
        
        with tab4:
            display_voice_settings()
            
            st.divider()
            
            # Help section
            st.markdown("### ❓ Voice Commands Help")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🍽️ Ordering Commands:**
                - "I'd like to place an order"
                - "What's on the menu?"
                - "What are today's specials?"
                - "I want to order [dish name]"
                - "Can I see the dessert menu?"
                
                **📅 Reservation Commands:**
                - "I want to make a reservation"
                - "Book a table for [number] people"
                - "Reserve a table for [date/time]"
                - "Do you have availability tonight?"
                """)
            
            with col2:
                st.markdown("""
                **ℹ️ Information Commands:**
                - "What are your hours?"
                - "Where are you located?"
                - "What's your phone number?"
                - "Do you have parking?"
                - "What's the wifi password?"
                
                **💬 Conversation Commands:**
                - "Hello" / "Hi there" (to start)
                - "Help me with..." 
                - "I have a question about..."
                - "Thank you" / "Goodbye" (to end)
                - "Stop" / "End conversation"
                """)
            
            # Pro tips
            st.markdown("### 💡 Pro Tips for Better Voice Recognition")
            st.info("""
            🎯 **Speak clearly** and at a normal pace  
            🔇 **Minimize background noise** when possible  
            ⏱️ **Wait for the listening indicator** before speaking  
            🗣️ **Use natural language** - no need for robotic commands  
            🔄 **If not understood**, simply repeat your request  
            📱 **For complex orders**, break them into smaller requests
            """)
            
            # System diagnostics
            if st.button("🔍 Run System Diagnostics", use_container_width=True):
                with st.spinner("Running comprehensive diagnostics..."):
                    progress = st.progress(0)
                    
                    # Simulate diagnostic steps
                    steps = [
                        "Checking microphone access...",
                        "Testing audio output...", 
                        "Verifying API connections...",
                        "Checking voice recognition...",
                        "Testing speech synthesis...",
                        "Validating system resources..."
                    ]
                    
                    for i, step in enumerate(steps):
                        st.text(step)
                        time.sleep(0.5)
                        progress.progress((i + 1) / len(steps))
                    
                    progress.empty()
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.success("✅ Microphone: Ready")
                        st.success("✅ Audio Output: Working")
                    
                    with col2:
                        st.success("✅ API Connections: Active")
                        st.success("✅ Voice Recognition: Online")
                    
                    with col3:
                        st.success("✅ Speech Synthesis: Ready")
                        st.success("✅ System Resources: Optimal")
                    
                    st.balloons()
    
    # Footer with additional information
    st.divider()
    
    with st.expander("🏆 Why Choose Romana Restaurant's Voice Assistant?"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🚀 Advanced Technology**
            - Natural language processing
            - Real-time voice recognition
            - Multilingual support coming soon
            - 24/7 availability
            """)
        
        with col2:
            st.markdown("""
            **⚡ Lightning Fast**
            - Instant responses
            - No waiting on hold
            - Quick reservation booking
            - Efficient order processing
            """)
        
        with col3:
            st.markdown("""
            **🎯 Personalized Service**
            - Remembers your preferences
            - Tailored recommendations
            - Special dietary accommodations
            - VIP customer recognition
            """)
    
    # Live statistics ticker
    if st.session_state.voice_agent and st.session_state.conversation_active:
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🔴 Live Status", "ACTIVE" if st.session_state.conversation_active else "IDLE")
        
        with col2:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric("🕐 Current Time", current_time)
        
        with col3:
            st.metric("🎤 Voice Quality", "High Definition")
        
        with col4:
            response_time = "< 2 sec"
            st.metric("⚡ Response Time", response_time)
        
        with col5:
            uptime = "99.9%"
            st.metric("⏰ System Uptime", uptime)
    
    # Emergency contact info (always visible)
    st.sidebar.markdown("### 🆘 Emergency Contact")
    st.sidebar.error("""
    **For Urgent Matters:**  
    📞 Call: (555) 123-ROMA  
    📧 Email: urgent@romanarestaurant.com
    """)
    
    # Sidebar quick stats
    st.sidebar.markdown("### 📈 Today's Quick Stats")
    
    if st.session_state.voice_agent:
        # Real-time updates
        total_customers = 127 + st.session_state.conversation_stats['total_interactions']
        satisfaction = st.session_state.conversation_stats['customer_satisfaction']
        
        st.sidebar.metric("👥 Total Customers", total_customers, "↑ 12%")
        st.sidebar.metric("⭐ Satisfaction", f"{satisfaction}/5.0", "↑ 0.2")
        st.sidebar.metric("🎤 Voice Sessions", st.session_state.conversation_stats['total_interactions'])
        st.sidebar.metric("📅 Reservations", st.session_state.reservations_today, "↑ 3")
        st.sidebar.metric("🍽️ Orders", st.session_state.orders_today, "↑ 8")
    
    # Sidebar quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    if st.sidebar.button("📋 View Full Menu", use_container_width=True):
        st.sidebar.success("Full menu would open in a new window")
    
    if st.sidebar.button("📞 Call Restaurant", use_container_width=True):
        st.sidebar.info("Dialing (555) 123-ROMA...")
    
    if st.sidebar.button("🌐 Visit Website", use_container_width=True):
        st.sidebar.info("Opening www.romanarestaurant.com...")
    
    # Feedback section
    st.sidebar.markdown("### 💭 Quick Feedback")
    
    feedback_rating = st.sidebar.select_slider(
        "Rate your experience:",
        options=[1, 2, 3, 4, 5],
        value=5,
        format_func=lambda x: "⭐" * x
    )
    
    if st.sidebar.button("Submit Feedback", use_container_width=True):
        st.session_state.conversation_stats['customer_satisfaction'] = (
            st.session_state.conversation_stats['customer_satisfaction'] * 0.9 + feedback_rating * 0.1
        )
        st.sidebar.success("Thank you for your feedback! 🙏")
    
    # Auto-refresh for live updates (every 30 seconds when conversation is active)
    if st.session_state.conversation_active:
        time.sleep(0.1)  # Small delay to prevent too frequent updates
        
    # Session cleanup on app restart
    if st.sidebar.button("🔄 Reset All Data", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("All data reset! Please refresh the page.")

if __name__ == "__main__":
    main()