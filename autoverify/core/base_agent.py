"""
Base Agent Class for AutoVerify System
=====================================

This module defines the base agent class that all AutoVerify agents inherit from.
It provides common functionality for agent communication, logging, and state management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentMessage:
    """Standard message format for agent communication"""
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    message_id: str

@dataclass
class VerificationResult:
    """Result of a verification process"""
    claim: str
    confidence_score: float  # 0-1 scale
    sources: List[str]
    verification_status: str  # "verified", "unverified", "contradicted"
    evidence: List[Dict[str, Any]]
    timestamp: datetime

@dataclass
class AgentState:
    """State management for agents"""
    agent_id: str
    status: str  # "idle", "processing", "error"
    current_task: Optional[str]
    processed_messages: int
    last_activity: datetime

class BaseAgent(ABC):
    """
    Base class for all AutoVerify agents.
    
    This class provides:
    - Standard message handling
    - State management
    - Logging and monitoring
    - Error handling
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState(
            agent_id=agent_id,
            status="idle",
            current_task=None,
            processed_messages=0,
            last_activity=datetime.now()
        )
        self.message_queue: List[AgentMessage] = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abstractmethod
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message and return a response.
        
        Args:
            message: The incoming message to process
            
        Returns:
            AgentMessage: The response message
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return a list of capabilities this agent provides.
        
        Returns:
            List[str]: List of capability descriptions
        """
        pass
    
    def send_message(self, recipient: str, message_type: str, content: Dict[str, Any]) -> str:
        """
        Send a message to another agent.
        
        Args:
            recipient: ID of the recipient agent
            message_type: Type of message (e.g., "verification_request", "correction_request")
            content: Message content
            
        Returns:
            str: Message ID for tracking
        """
        message_id = f"{self.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            message_id=message_id
        )
        
        self.logger.info(f"Sending {message_type} to {recipient}")
        return message_id
    
    def receive_message(self, message: AgentMessage) -> None:
        """
        Receive and queue a message for processing.
        
        Args:
            message: The incoming message
        """
        self.message_queue.append(message)
        self.logger.info(f"Received {message.message_type} from {message.sender}")
    
    def update_state(self, status: str, task: Optional[str] = None) -> None:
        """
        Update the agent's state.
        
        Args:
            status: New status ("idle", "processing", "error")
            task: Current task description
        """
        self.state.status = status
        self.state.current_task = task
        self.state.last_activity = datetime.now()
        
        if status == "processing":
            self.state.processed_messages += 1
            
        self.logger.info(f"State updated: {status} - {task}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status and statistics.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.state.status,
            "current_task": self.state.current_task,
            "processed_messages": self.state.processed_messages,
            "last_activity": self.state.last_activity.isoformat(),
            "queue_size": len(self.message_queue),
            "capabilities": self.get_capabilities()
        }
    
    def handle_error(self, error: Exception, context: str = "") -> None:
        """
        Handle errors with proper logging and state management.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
        """
        self.update_state("error", f"Error in {context}: {str(error)}")
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def reset_state(self) -> None:
        """Reset agent state to idle."""
        self.update_state("idle")
        self.message_queue.clear()
        self.logger.info("Agent state reset")
