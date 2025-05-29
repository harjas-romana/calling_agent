import React, { useState, useEffect, useRef } from 'react';
import { 
  Send,
  MessageCircle, 
  Calendar, 
  ShoppingCart,  
  Clock, 
  Star,
  Volume2,
  VolumeX,
  Loader2,
  ChefHat,
  User,
  Bot,
  Info,
  PhoneCall,
  Map,
  Clock3
} from 'lucide-react';
import { VoiceInterface } from './VoiceInterface';
import { ReservationModal } from './ReservationModal';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  audioUrl?: string;
}

interface QuickAction {
  id: string;
  icon: React.ReactNode;
  label: string;
  description: string;
  action: string;
  gradient: string;
}

const App: React.FC = () => {
  // State management
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [showReservationModal, setShowReservationModal] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<any[]>([]);
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Mock API endpoints (replace with your actual backend)
  const API_BASE = 'http://localhost:8000';
  
  // Quick actions configuration
  const quickActions: QuickAction[] = [
    {
      id: 'reservation',
      icon: <Calendar className="w-5 h-5" />,
      label: 'Make Reservation',
      description: 'Book a table for your dining experience',
      action: 'I would like to make a reservation',
      gradient: 'from-blue-600 to-blue-800'
    },
    {
      id: 'menu',
      icon: <ChefHat className="w-5 h-5" />,
      label: 'View Menu',
      description: 'Explore our delicious Italian dishes',
      action: 'Can you show me the menu and today\'s specials?',
      gradient: 'from-cyan-600 to-blue-700'
    },
    {
      id: 'order',
      icon: <ShoppingCart className="w-5 h-5" />,
      label: 'Place Order',
      description: 'Order food for pickup or delivery',
      action: 'I would like to place an order',
      gradient: 'from-indigo-600 to-blue-800'
    },
    {
      id: 'hours',
      icon: <Clock3 className="w-5 h-5" />,
      label: 'Hours & Location',
      description: 'Find our location and operating hours',
      action: 'What are your hours and where are you located?',
      gradient: 'from-sky-600 to-blue-700'
    }
  ];

  // Welcome message
  useEffect(() => {
    const welcomeMessage: Message = {
      id: '1',
      type: 'assistant',
      content: 'Welcome to Romana Restaurant! I\'m your AI assistant. How can I help you today? You can make reservations, place orders, or ask about our menu and services.',
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  }, []);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle sending messages
  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString() + '_user',
      type: 'user',
      content: content.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Simulate API call to your voice agent backend
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content,
          conversation_history: conversationHistory
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      
      const assistantMessage: Message = {
        id: Date.now().toString() + '_assistant',
        type: 'assistant',
        content: data.response || 'I apologize, but I encountered an error processing your request.',
        timestamp: new Date(),
        audioUrl: data.audio_url
      };

      setMessages(prev => [...prev, assistantMessage]);
      setConversationHistory(data.conversation_history || []);

      // Play audio if available and not muted
      if (data.audio_url && !isMuted && audioRef.current) {
        audioRef.current.src = data.audio_url;
        audioRef.current.play().catch(console.error);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: Message = {
        id: Date.now().toString() + '_error',
        type: 'assistant',
        content: 'I\'m sorry, I\'m having trouble connecting right now. Please try again in a moment.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle quick actions
  const handleQuickAction = (action: string) => {
    if (action.includes('reservation')) {
      setShowReservationModal(true);
    } else {
      handleSendMessage(action);
    }
  };

  // Handle voice input
  const handleVoiceInput = (transcript: string) => {
    if (transcript) {
      handleSendMessage(transcript);
    }
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSendMessage(inputMessage);
  };

  // Format timestamp
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true 
    });
  };

  return (
    <div className="min-h-screen bg-black text-gray-100">
      {/* Header */}
      <header className="bg-gradient-to-r from-black to-gray-900 border-b border-gray-800 shadow-xl">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-blue-800 rounded-lg flex items-center justify-center shadow-lg">
                <ChefHat className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white tracking-tight">Romana Restaurant</h1>
                <p className="text-xs text-blue-300 font-mono">AI Voice Assistant</p>
              </div>
            </div>

            
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setIsMuted(!isMuted)}
                className={`p-2 rounded-lg transition-all duration-200 hover:bg-gray-800 hover:shadow-md ${
                  isMuted 
                    ? 'text-red-400' 
                    : 'text-blue-400'
                }`}
                aria-label={isMuted ? "Unmute" : "Mute"}
              >
                {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
              </button>
              
              <div className="hidden md:flex items-center space-x-2 text-xs text-blue-300 bg-gray-900/50 px-3 py-1.5 rounded-full border border-gray-800">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <span className="font-mono">SYSTEM ONLINE</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Quick Actions Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 shadow-xl">
              <h2 className="text-lg font-bold text-white mb-5 flex items-center">
                <Star className="w-5 h-5 mr-2 text-blue-400" />
                Quick Actions
              </h2>
              
              <div className="space-y-3">
                {quickActions.map((action) => (
                  <button
                    key={action.id}
                    onClick={() => handleQuickAction(action.action)}
                    className={`w-full p-3 rounded-xl bg-gradient-to-r ${action.gradient} hover:scale-[1.02] transform transition-all duration-200 shadow-lg hover:shadow-blue-500/20 flex items-center space-x-3 group`}
                  >
                    <div className="bg-black/20 p-2 rounded-lg group-hover:bg-black/30 transition-colors">
                      {action.icon}
                    </div>
                    <div className="text-left">
                      <div className="font-medium text-white text-sm">{action.label}</div>
                      <div className="text-xs text-blue-100 opacity-80">{action.description}</div>
                    </div>
                  </button>
                ))}
              </div>

              {/* Restaurant Info */}
              <div className="mt-6 pt-6 border-t border-gray-800">
                <h3 className="text-xs font-semibold text-blue-300 uppercase tracking-wider mb-3 flex items-center">
                  <Info className="w-4 h-4 mr-1" />
                  Contact Info
                </h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-center space-x-2 text-blue-100 hover:text-white transition-colors">
                    <div className="bg-blue-900/50 p-1.5 rounded-lg">
                      <PhoneCall className="w-4 h-4" />
                    </div>
                    <span>(555) 123-4567</span>
                  </div>
                  <div className="flex items-center space-x-2 text-blue-100 hover:text-white transition-colors">
                    <div className="bg-blue-900/50 p-1.5 rounded-lg">
                      <Map className="w-4 h-4" />
                    </div>
                    <span>123 Pasta St, Toronto</span>
                  </div>
                  <div className="flex items-center space-x-2 text-blue-100 hover:text-white transition-colors">
                    <div className="bg-blue-900/50 p-1.5 rounded-lg">
                      <Clock className="w-4 h-4" />
                    </div>
                    <span>11AM - 10PM Daily</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Chat Interface */}
          <div className="lg:col-span-3">
            <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden shadow-xl">
              {/* Messages Area */}
              <div className="h-[500px] overflow-y-auto p-6 space-y-6 bg-gradient-to-b from-gray-900/80 to-gray-900/20">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
                      <div
                        className={`rounded-xl px-4 py-3 shadow-lg ${
                          message.type === 'user'
                            ? 'bg-gradient-to-r from-blue-600 to-blue-800 text-white'
                            : 'bg-gray-800 text-gray-100 border border-gray-700'
                        }`}
                      >
                        <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                        <div className={`text-xs mt-2 flex justify-end items-center ${
                          message.type === 'user' ? 'text-blue-200' : 'text-gray-400'
                        }`}>
                          {formatTime(message.timestamp)}
                        </div>
                      </div>
                    </div>
                    
                    <div className={`w-9 h-9 rounded-full flex items-center justify-center shadow-md ${
                      message.type === 'user' 
                        ? 'order-1 mr-3 bg-gradient-to-br from-blue-700 to-blue-900' 
                        : 'order-2 ml-3 bg-gradient-to-br from-gray-700 to-gray-900 border border-gray-700'
                    }`}>
                      {message.type === 'user' ? (
                        <User className="w-4 h-4 text-white" />
                      ) : (
                        <Bot className="w-4 h-4 text-white" />
                      )}
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="flex items-center space-x-3">
                    <div className="w-9 h-9 rounded-full bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center shadow-md border border-gray-700">
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div className="bg-gray-800 rounded-xl px-4 py-3 border border-gray-700 shadow-md">
                      <div className="flex items-center space-x-2 text-sm text-blue-300">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Processing your request...</span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="border-t border-gray-800 p-4 bg-gray-900/50 backdrop-blur-sm">
                <form onSubmit={handleSubmit} className="flex items-center space-x-3">
                  <div className="flex-1 relative">
                    <input
                      ref={inputRef}
                      type="text"
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      placeholder="Type your message or use voice input..."
                      className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent transition-all duration-200 shadow-inner"
                      disabled={isLoading}
                    />
                  </div>
                  
                  <VoiceInterface
                    onVoiceInput={handleVoiceInput}
                    isListening={isListening}
                    setIsListening={setIsListening}
                    disabled={isLoading}
                  />
                  
                  <button
                    type="submit"
                    disabled={isLoading || !inputMessage.trim()}
                    className="p-3 bg-gradient-to-r from-blue-600 to-blue-800 text-white rounded-xl hover:from-blue-700 hover:to-blue-900 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:shadow-lg hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
                  >
                    {isLoading ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <Send className="w-5 h-5" />
                    )}
                  </button>
                </form>

                <div className="flex items-center justify-center mt-3 text-xs text-gray-500">
                  <MessageCircle className="w-3 h-3 mr-1.5" />
                  <span>Press Enter to send, or click the microphone for voice input</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Reservation Modal */}
      {showReservationModal && (
        <ReservationModal
          isOpen={showReservationModal}
          onClose={() => setShowReservationModal(false)}
          onSubmit={(reservation) => {
            console.log('Reservation submitted:', reservation);
            setShowReservationModal(false);
            handleSendMessage(`I would like to make a reservation for ${reservation.partySize} people on ${reservation.date} at ${reservation.time} under the name ${reservation.name}. My phone number is ${reservation.phone}.`);
          }}
        />
      )}

      {/* Hidden audio element for TTS playback */}
      <audio ref={audioRef} />
    </div>
  );
};

export default App;