"""
Generator Agent - LLM Core
=========================

The Generator Agent is responsible for:
1. Taking user queries and producing initial responses
2. Using various LLM models (GPT-4, Claude, Llama, etc.)
3. Passing responses to the verification layer
4. Managing model selection and configuration
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Optional import for Anthropic
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ..core.base_agent import BaseAgent, AgentMessage, AgentState

class GeneratorAgent(BaseAgent):
    """
    Generator Agent for producing initial LLM responses.
    
    This agent handles:
    - Model selection and configuration
    - Prompt engineering
    - Response generation
    - Quality assessment
    """
    
    def __init__(self, agent_id: str = "generator", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "generator")
        
        # Default model configuration
        self.model_config = config or {
            "primary_model": "gpt-4",
            "fallback_model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        # Initialize models
        self.models = self._initialize_models()
        self.current_model = self.models[self.model_config["primary_model"]]
        
        # Response templates
        self.response_templates = {
            "standard": "You are a helpful AI assistant. Provide accurate, well-reasoned responses.",
            "factual": "You are a factual AI assistant. Focus on accuracy and cite sources when possible.",
            "creative": "You are a creative AI assistant. Be imaginative while maintaining factual accuracy."
        }
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize available LLM models."""
        models = {}
        
        # OpenAI Models
        try:
            models["gpt-4"] = ChatOpenAI(
                model="gpt-4",
                temperature=self.model_config["temperature"],
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.logger.info("GPT-4 model initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing GPT-4: {e}")
        
        try:
            models["gpt-3.5-turbo"] = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=self.model_config["temperature"],
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.logger.info("GPT-3.5-turbo model initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing GPT-3.5-turbo: {e}")
        
        # Anthropic Models (if available)
        if ANTHROPIC_AVAILABLE:
            try:
                models["claude-3"] = ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    temperature=self.model_config["temperature"]
                )
                self.logger.info("Claude-3 model initialized successfully")
            except Exception as e:
                self.logger.warning(f"Claude model not available: {e}")
        else:
            self.logger.info("Anthropic models not available (langchain-anthropic not installed)")
            
        if not models:
            raise RuntimeError("No models could be initialized. Please check your API keys and configuration.")
            
        return models
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process a generation request and produce an initial response.
        
        Args:
            message: Message containing user query and generation parameters
            
        Returns:
            AgentMessage: Response with generated content
        """
        try:
            self.update_state("processing", "Generating response")
            
            # Extract request details
            user_query = message.content.get("query", "")
            context = message.content.get("context", "")
            response_style = message.content.get("style", "standard")
            model_preference = message.content.get("model", self.model_config["primary_model"])
            
            # Generate response
            response = self._generate_response(
                query=user_query,
                context=context,
                style=response_style,
                model_name=model_preference
            )
            
            # Create response message
            response_content = {
                "generated_text": response["text"],
                "model_used": response["model"],
                "generation_metadata": response["metadata"],
                "confidence_indicators": response["confidence"],
                "timestamp": datetime.now().isoformat()
            }
            
            self.update_state("idle")
            
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type="generation_response",
                content=response_content,
                timestamp=datetime.now(),
                message_id=f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            
        except Exception as e:
            self.handle_error(e, "response generation")
            return self._create_error_response(message.sender, str(e))
    
    def _generate_response(self, query: str, context: str = "", style: str = "standard", model_name: str = None) -> Dict[str, Any]:
        """
        Generate a response using the specified model and style.
        
        Args:
            query: User's question or request
            context: Additional context for the query
            style: Response style (standard, factual, creative)
            model_name: Specific model to use
            
        Returns:
            Dict containing generated text and metadata
        """
        # Select model
        model = self.models.get(model_name, self.current_model)
        
        # Prepare prompt
        system_prompt = self.response_templates.get(style, self.response_templates["standard"])
        
        if context:
            system_prompt += f"\n\nContext: {context}"
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        # Generate response
        try:
            response = model.invoke(messages)
            
            # Extract response details
            generated_text = response.content
            
            # Calculate confidence indicators (simplified)
            confidence_indicators = self._calculate_confidence_indicators(generated_text, query)
            
            return {
                "text": generated_text,
                "model": model_name or self.model_config["primary_model"],
                "metadata": {
                    "style": style,
                    "context_provided": bool(context),
                    "response_length": len(generated_text),
                    "generation_time": datetime.now().isoformat()
                },
                "confidence": confidence_indicators
            }
            
        except Exception as e:
            # Fallback to secondary model
            if model_name != self.model_config["fallback_model"]:
                self.logger.warning(f"Primary model failed, trying fallback: {e}")
                return self._generate_response(query, context, style, self.model_config["fallback_model"])
            else:
                raise e
    
    def _calculate_confidence_indicators(self, response: str, query: str) -> Dict[str, float]:
        """
        Calculate confidence indicators for the generated response.
        
        Args:
            response: Generated response text
            query: Original user query
            
        Returns:
            Dict with confidence scores
        """
        # Simplified confidence calculation
        # In a real implementation, this would be more sophisticated
        
        indicators = {
            "response_completeness": min(1.0, len(response) / 100),  # Longer responses often more complete
            "query_relevance": 0.8,  # Placeholder - would use semantic similarity
            "factual_claims": 0.7,   # Placeholder - would analyze for factual statements
            "citation_presence": 0.3 if any(word in response.lower() for word in ["source", "reference", "according to"]) else 0.0
        }
        
        # Overall confidence (weighted average)
        indicators["overall_confidence"] = sum(indicators.values()) / len(indicators)
        
        return indicators
    
    def _create_error_response(self, recipient: str, error_message: str) -> AgentMessage:
        """Create an error response message."""
        return AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type="generation_error",
            content={
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            message_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "Text generation using multiple LLM models",
            "Response style customization",
            "Context-aware generation",
            "Confidence scoring",
            "Model fallback handling",
            "Prompt engineering"
        ]
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_name in self.models:
            self.current_model = self.models[model_name]
            self.logger.info(f"Switched to model: {model_name}")
            return True
        else:
            self.logger.error(f"Model not available: {model_name}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def update_model_config(self, config: Dict[str, Any]) -> None:
        """
        Update model configuration.
        
        Args:
            config: New configuration parameters
        """
        self.model_config.update(config)
        self.logger.info(f"Model configuration updated: {config}")
