import os
import json
import base64
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Import your existing classes
from rag_layer import RAGLayer
from voice_agent import VoiceAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for our agents
rag_layer: Optional[RAGLayer] = None
voice_agent: Optional[VoiceAgent] = None

# Pydantic models for API requests/responses
class ChatMessage(BaseModel):
    message: str = Field(..., description="User's message")
    conversation_history: List[Dict[str, str]] = Field(default=[], description="Previous conversation")
    use_voice: bool = Field(default=False, description="Whether to generate voice response")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant's response")
    conversation_history: List[Dict[str, str]] = Field(..., description="Updated conversation history")
    audio_url: Optional[str] = Field(None, description="URL to audio file if voice response generated")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")

class ReservationRequest(BaseModel):
    name: str = Field(..., description="Customer name")
    phone: str = Field(..., description="Phone number")
    email: str = Field(..., description="Email address")
    party_size: int = Field(..., ge=1, le=12, description="Number of people")
    date: str = Field(..., description="Reservation date (YYYY-MM-DD)")
    time: str = Field(..., description="Reservation time (HH:MM AM/PM)")
    special_requests: Optional[str] = Field(None, description="Special requests or dietary restrictions")

class OrderRequest(BaseModel):
    table_number: int = Field(..., ge=1, description="Table number")
    items: List[str] = Field(..., description="List of menu items")
    special_requests: Optional[str] = Field(None, description="Special preparation requests")

class KnowledgeBaseUpdate(BaseModel):
    section: Optional[str] = Field(None, description="Section to update (e.g., 'restaurant_info/hours')")
    data: Dict[str, Any] = Field(..., description="Data to update")
    merge: bool = Field(True, description="Whether to merge with existing data")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    rag_layer_status: str = Field(..., description="RAG Layer status")
    voice_agent_status: str = Field(..., description="Voice Agent status")

# Application startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global rag_layer, voice_agent
    
    try:
        # Initialize components
        logger.info("Initializing AI components...")
        
        # Get API keys from environment variables
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "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")
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "sk_a643471cf3d2de658ac47648b33d8314bfe39dcc14ebfe7b")
        
        if not openrouter_key or not elevenlabs_key:
            raise ValueError("Missing required API keys. Please set OPENROUTER_API_KEY and ELEVENLABS_API_KEY environment variables.")
        
        # Initialize RAG Layer
        rag_layer = RAGLayer(
            api_key=openrouter_key,
            model="deepseek/deepseek-prover-v2:free",
            conversation_memory=10
        )
        
        # Initialize Voice Agent
        voice_agent = VoiceAgent(
            elevenlabs_api_key=elevenlabs_key,
            rag_layer=rag_layer,
            voice_id="21m00Tcm4TlvDq8ikWAM"  # Rachel's voice
        )
        
        logger.info("AI components initialized successfully!")
        
        # Create necessary directories
        os.makedirs("static/audio", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize AI components: {str(e)}")
        raise
    finally:
        logger.info("Shutting down AI components...")

# Create FastAPI app
app = FastAPI(
    title="Romana Restaurant AI Voice Agent API",
    description="Backend API for Romana Restaurant's AI Voice Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for audio
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        rag_layer_status="active" if rag_layer else "inactive",
        voice_agent_status="active" if voice_agent else "inactive"
    )

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    """Main chat endpoint for interacting with the AI assistant"""
    try:
        if not voice_agent:
            raise HTTPException(status_code=503, detail="Voice agent not initialized")
        
        logger.info(f"Processing chat request: {request.message[:50]}...")
        
        # Handle conversation with voice agent
        response_text, audio_data, updated_history = voice_agent.handle_conversation(
            query=request.message,
            conversation_history=request.conversation_history
        )
        
        # Prepare response
        chat_response = ChatResponse(
            response=response_text,
            conversation_history=updated_history
        )
        
        # Handle audio response if requested or generated
        if audio_data and request.use_voice:
            try:
                # Save audio file
                audio_filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp3"
                audio_path = f"static/audio/{audio_filename}"
                
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                
                chat_response.audio_url = f"/static/audio/{audio_filename}"
                
                # Also provide base64 encoded audio for direct playback
                chat_response.audio_data = base64.b64encode(audio_data).decode('utf-8')
                
            except Exception as audio_error:
                logger.error(f"Error processing audio: {str(audio_error)}")
                # Continue without audio
        
        return chat_response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Voice-only endpoint for direct audio communication
@app.post("/voice-chat")
async def voice_chat(request: ChatMessage):
    """Voice-focused chat endpoint that always returns audio"""
    try:
        if not voice_agent:
            raise HTTPException(status_code=503, detail="Voice agent not initialized")
        
        # Force voice response
        request.use_voice = True
        return await chat(request)
        
    except Exception as e:
        logger.error(f"Error in voice chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Text-to-Speech endpoint
@app.post("/tts")
async def text_to_speech(text: str):
    """Convert text to speech"""
    try:
        if not voice_agent:
            raise HTTPException(status_code=503, detail="Voice agent not initialized")
        
        audio_data = voice_agent.text_to_speech(text)
        
        if audio_data:
            # Return as streaming response
            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=tts_audio.mp3"}
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
            
    except Exception as e:
        logger.error(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Reservations endpoint
@app.post("/reservations")
async def create_reservation(reservation: ReservationRequest):
    """Create a new reservation"""
    try:
        if not voice_agent:
            raise HTTPException(status_code=503, detail="Voice agent not initialized")
        
        # Format reservation message for the voice agent
        reservation_message = (
            f"I would like to make a reservation for {reservation.party_size} people "
            f"on {reservation.date} at {reservation.time}. "
            f"The name is {reservation.name}, phone number {reservation.phone}, "
            f"and email {reservation.email}."
        )
        
        if reservation.special_requests:
            reservation_message += f" Special requests: {reservation.special_requests}"
        
        # Process through voice agent
        response_text, audio_data, conversation_history = voice_agent.handle_conversation(
            query=reservation_message,
            conversation_history=[]
        )
        
        # Store reservation (in a real app, you'd save to database)
        reservation_data = reservation.dict()
        reservation_data["created_at"] = datetime.now().isoformat()
        reservation_data["status"] = "confirmed"
        
        # Add to voice agent's reservations
        voice_agent.reservations.append(reservation_data)
        
        return {
            "status": "success",
            "message": "Reservation created successfully",
            "reservation_id": len(voice_agent.reservations),
            "confirmation": response_text,
            "reservation_data": reservation_data
        }
        
    except Exception as e:
        logger.error(f"Error creating reservation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Orders endpoint
@app.post("/orders")
async def create_order(order: OrderRequest):
    """Create a new food order"""
    try:
        if not voice_agent:
            raise HTTPException(status_code=503, detail="Voice agent not initialized")
        
        # Format order message
        items_text = ", ".join(order.items)
        order_message = (
            f"I would like to place an order for table {order.table_number}. "
            f"Items: {items_text}."
        )
        
        if order.special_requests:
            order_message += f" Special requests: {order.special_requests}"
        
        # Process through voice agent
        response_text, audio_data, conversation_history = voice_agent.handle_conversation(
            query=order_message,
            conversation_history=[]
        )
        
        # Store order
        order_data = order.dict()
        order_data["created_at"] = datetime.now().isoformat()
        order_data["status"] = "received"
        order_data["estimated_time"] = "25-30 minutes"
        
        # Add to voice agent's orders
        voice_agent.orders.append(order_data)
        
        return {
            "status": "success",
            "message": "Order placed successfully",
            "order_id": len(voice_agent.orders),
            "confirmation": response_text,
            "order_data": order_data
        }
        
    except Exception as e:
        logger.error(f"Error creating order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge base endpoints
@app.get("/knowledge-base")
async def get_knowledge_base(section: Optional[str] = None):
    """Get knowledge base information"""
    try:
        if not rag_layer:
            raise HTTPException(status_code=503, detail="RAG layer not initialized")
        
        data = rag_layer.get_knowledge_base(section)
        return {"status": "success", "data": data}
        
    except Exception as e:
        logger.error(f"Error getting knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base")
async def update_knowledge_base(update: KnowledgeBaseUpdate):
    """Update knowledge base information"""
    try:
        if not rag_layer:
            raise HTTPException(status_code=503, detail="RAG layer not initialized")
        
        success = rag_layer.update_knowledge_base(
            new_data=update.data,
            section=update.section,
            merge=update.merge
        )
        
        if success:
            return {"status": "success", "message": "Knowledge base updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update knowledge base")
            
    except Exception as e:
        logger.error(f"Error updating knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Menu endpoint
@app.get("/menu")
async def get_menu():
    """Get restaurant menu"""
    try:
        if not rag_layer:
            raise HTTPException(status_code=503, detail="RAG layer not initialized")
        
        menu_data = rag_layer.get_knowledge_base("restaurant_info/popular_dishes")
        specials = rag_layer.get_knowledge_base("restaurant_info/specials")
        
        return {
            "status": "success",
            "menu": menu_data,
            "specials": specials,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting menu: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Restaurant info endpoint
@app.get("/restaurant-info")
async def get_restaurant_info():
    """Get restaurant information"""
    try:
        if not rag_layer:
            raise HTTPException(status_code=503, detail="RAG layer not initialized")
        
        info = rag_layer.get_knowledge_base("restaurant_info")
        return {"status": "success", "restaurant_info": info}
        
    except Exception as e:
        logger.error(f"Error getting restaurant info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoint
@app.get("/analytics")
async def get_analytics():
    """Get basic analytics"""
    try:
        if not voice_agent:
            raise HTTPException(status_code=503, detail="Voice agent not initialized")
        
        return {
            "status": "success",
            "analytics": {
                "total_reservations": len(voice_agent.reservations),
                "total_orders": len(voice_agent.orders),
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Backup endpoints
@app.post("/backup")
async def create_backup():
    """Create a backup of the knowledge base"""
    try:
        if not rag_layer:
            raise HTTPException(status_code=503, detail="RAG layer not initialized")
        
        filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        success = rag_layer.backup_knowledge_base(filename)
        
        if success:
            return {"status": "success", "message": f"Backup created: {filename}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create backup")
            
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )

# Background task for cleanup
async def cleanup_old_audio_files():
    """Clean up old audio files to save disk space"""
    try:
        audio_dir = "static/audio"
        if os.path.exists(audio_dir):
            now = datetime.now()
            for filename in os.listdir(audio_dir):
                filepath = os.path.join(audio_dir, filename)
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    if (now - file_time).days > 1:  # Delete files older than 1 day
                        os.remove(filepath)
                        logger.info(f"Cleaned up old audio file: {filename}")
    except Exception as e:
        logger.error(f"Error cleaning up audio files: {str(e)}")

# Main entry point
if __name__ == "__main__":
    # Set up environment variables if not already set
    if not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = "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"
    
    if not os.getenv("ELEVENLABS_API_KEY"):
        os.environ["ELEVENLABS_API_KEY"] = "sk_a643471cf3d2de658ac47648b33d8314bfe39dcc14ebfe7b"
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )