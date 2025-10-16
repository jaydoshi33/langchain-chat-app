"""
AutoVerify Orchestrator - Multi-Agent Coordination
================================================

The Orchestrator coordinates all AutoVerify agents to provide
end-to-end hallucination detection and fact verification.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json

from .base_agent import BaseAgent, AgentMessage
from ..agents.generator_agent import GeneratorAgent
from ..agents.verifier_agent import VerifierAgent
from ..agents.correction_agent import CorrectionAgent
from ..agents.audit_agent import AuditAgent

class AutoVerifyOrchestrator:
    """
    Main orchestrator for the AutoVerify system.
    
    This class coordinates all agents to provide:
    - End-to-end verification pipeline
    - Agent communication and coordination
    - Error handling and fallback mechanisms
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.session_id = None
        
        # Initialize agents
        self.agents = {
            "generator": GeneratorAgent(config=self.config.get("generator", {})),
            "verifier": VerifierAgent(config=self.config.get("verifier", {})),
            "corrector": CorrectionAgent(config=self.config.get("corrector", {})),
            "auditor": AuditAgent(config=self.config.get("auditor", {}))
        }
        
        # Agent communication registry
        self.message_registry = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0
        }
    
    async def process_query(self, query: str, context: Dict[str, Any] = None, 
                          verification_level: str = "standard") -> Dict[str, Any]:
        """
        Process a user query through the complete AutoVerify pipeline.
        
        Args:
            query: User's question or request
            context: Additional context for the query
            verification_level: Level of verification (basic, standard, thorough)
            
        Returns:
            Complete verification result with corrections and audit report
        """
        start_time = datetime.now()
        self.session_id = f"session_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            self.performance_metrics["total_requests"] += 1
            
            # Step 1: Generate initial response
            generated_response = await self._generate_response(query, context or {})
            
            # Step 2: Verify the generated response
            verification_results = await self._verify_response(
                generated_response["generated_text"], 
                verification_level
            )
            
            # Step 3: Apply corrections if needed
            correction_results = await self._apply_corrections(
                generated_response["generated_text"],
                verification_results
            )
            
            # Step 4: Generate audit report
            audit_report = await self._generate_audit_report(
                query, generated_response, verification_results, correction_results
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics["successful_requests"] += 1
            self._update_average_processing_time(processing_time)
            
            # Compile final result
            result = {
                "session_id": self.session_id,
                "query": query,
                "original_response": generated_response["generated_text"],
                "final_response": correction_results.get("corrected_text", generated_response["generated_text"]),
                "generated_response": generated_response,
                "verification_results": verification_results,
                "correction_results": correction_results,
                "audit_report": audit_report,
                "processing_metadata": {
                    "processing_time": processing_time,
                    "verification_level": verification_level,
                    "timestamp": start_time.isoformat()
                }
            }
            
            return result
            
        except Exception as e:
            self.performance_metrics["failed_requests"] += 1
            return {
                "session_id": self.session_id,
                "error": str(e),
                "processing_metadata": {
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat()
                }
            }
    
    async def _generate_response(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial response using the Generator Agent."""
        try:
            # Create message for generator
            message = AgentMessage(
                sender="orchestrator",
                recipient="generator",
                message_type="generation_request",
                content={
                    "query": query,
                    "context": context,
                    "style": "factual"  # Use factual style for verification
                },
                timestamp=datetime.now(),
                message_id=f"gen_req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            
            # Process with generator agent
            response = self.agents["generator"].process_message(message)
            
            # Extract response content
            return response.content
            
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")
    
    async def _verify_response(self, text: str, verification_level: str) -> List[Dict[str, Any]]:
        """Verify the generated response using the Verifier Agent."""
        try:
            # Create message for verifier
            message = AgentMessage(
                sender="orchestrator",
                recipient="verifier",
                message_type="verification_request",
                content={
                    "text": text,
                    "level": verification_level,
                    "context": {}
                },
                timestamp=datetime.now(),
                message_id=f"verify_req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            
            # Process with verifier agent
            response = self.agents["verifier"].process_message(message)
            
            # Extract verification results
            return response.content.get("verification_results", [])
            
        except Exception as e:
            raise Exception(f"Verification failed: {str(e)}")
    
    async def _apply_corrections(self, original_text: str, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply corrections using the Correction Agent."""
        try:
            # Check if corrections are needed
            needs_correction = any(
                result.get("verification_status") in ["unverified", "contradicted"] or
                result.get("confidence_score", 1.0) < 0.7
                for result in verification_results
            )
            
            if not needs_correction:
                return {
                    "original_text": original_text,
                    "corrected_text": original_text,
                    "corrections_made": [],
                    "confidence_improvement": 0.0
                }
            
            # Create message for corrector
            message = AgentMessage(
                sender="orchestrator",
                recipient="corrector",
                message_type="correction_request",
                content={
                    "text": original_text,
                    "verification_results": verification_results,
                    "correction_type": "comprehensive"
                },
                timestamp=datetime.now(),
                message_id=f"correct_req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            
            # Process with correction agent
            response = self.agents["corrector"].process_message(message)
            
            # Extract correction results
            return response.content
            
        except Exception as e:
            raise Exception(f"Correction failed: {str(e)}")
    
    async def _generate_audit_report(self, query: str, generated_response: Dict[str, Any], 
                                   verification_results: List[Dict[str, Any]], 
                                   correction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audit report using the Audit Agent."""
        try:
            # Prepare session data
            session_data = {
                "user_id": "user",  # Would be provided in real implementation
                "query": query,
                "overall_confidence": sum(r.get("confidence_score", 0.0) for r in verification_results) / max(len(verification_results), 1),
                "hallucination_risk": self._assess_hallucination_risk(verification_results),
                "processing_time": 0.0,  # Will be updated by orchestrator
                "model_used": generated_response.get("model_used", "unknown")
            }
            
            # Create message for auditor
            message = AgentMessage(
                sender="orchestrator",
                recipient="auditor",
                message_type="audit_request",
                content={
                    "session_data": session_data,
                    "verification_results": verification_results,
                    "correction_results": correction_results,
                    "report_type": "comprehensive"
                },
                timestamp=datetime.now(),
                message_id=f"audit_req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            
            # Process with audit agent
            response = self.agents["auditor"].process_message(message)
            
            # Extract audit report
            return response.content
            
        except Exception as e:
            raise Exception(f"Audit report generation failed: {str(e)}")
    
    def _assess_hallucination_risk(self, verification_results: List[Dict[str, Any]]) -> str:
        """Assess overall hallucination risk based on verification results."""
        if not verification_results:
            return "unknown"
        
        contradicted_count = len([r for r in verification_results if r.get("verification_status") == "contradicted"])
        unverified_count = len([r for r in verification_results if r.get("verification_status") == "unverified"])
        total_count = len(verification_results)
        
        if contradicted_count > 0:
            return "high"
        elif unverified_count > total_count * 0.5:
            return "medium"
        else:
            return "low"
    
    def _update_average_processing_time(self, processing_time: float) -> None:
        """Update average processing time metric."""
        total_requests = self.performance_metrics["total_requests"]
        current_avg = self.performance_metrics["average_processing_time"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.performance_metrics["average_processing_time"] = new_avg
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health."""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_status()
        
        return {
            "system_status": "operational",
            "agents": agent_statuses,
            "performance_metrics": self.performance_metrics,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents."""
        capabilities = {}
        for agent_id, agent in self.agents.items():
            capabilities[agent_id] = agent.get_capabilities()
        return capabilities
    
    def reset_system(self) -> None:
        """Reset the system state."""
        self.session_id = None
        self.message_registry.clear()
        
        # Reset agent states
        for agent in self.agents.values():
            agent.reset_state()
        
        # Reset performance metrics
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0
        }
    
    def configure_agent(self, agent_id: str, config: Dict[str, Any]) -> bool:
        """
        Configure a specific agent.
        
        Args:
            agent_id: ID of the agent to configure
            config: Configuration parameters
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            return False
        
        try:
            # Update agent configuration
            if hasattr(self.agents[agent_id], 'config'):
                self.agents[agent_id].config.update(config)
            
            return True
        except Exception as e:
            print(f"Error configuring agent {agent_id}: {e}")
            return False
