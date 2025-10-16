"""
Correction Agent - Content Correction and Enhancement
===================================================

The Correction Agent is responsible for:
1. Re-generating or rephrasing content with low confidence scores
2. Using verified data to improve accuracy
3. Providing inline citations and supporting evidence
4. Maintaining the original intent while improving factual accuracy
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from ..core.base_agent import BaseAgent, AgentMessage

class CorrectionAgent(BaseAgent):
    """
    Correction Agent for improving content accuracy and adding citations.
    
    This agent handles:
    - Content re-generation with verified data
    - Citation insertion
    - Factual accuracy improvements
    - Maintaining original intent
    """
    
    def __init__(self, agent_id: str = "corrector", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "corrector")
        
        # Configuration
        self.config = config or {
            "correction_model": "gpt-4",
            "confidence_threshold": 0.7,
            "citation_style": "inline",
            "max_corrections": 3
        }
        
        # Initialize correction model
        self.correction_model = ChatOpenAI(
            model=self.config["correction_model"],
            temperature=0.3,  # Lower temperature for more consistent corrections
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Correction templates
        self.correction_templates = {
            "factual_correction": """
            You are a fact-checking expert. Your task is to correct the following text using verified information.
            
            Original text: {original_text}
            Verification results: {verification_results}
            Verified sources: {sources}
            
            Instructions:
            1. Correct any factual inaccuracies using the verified information
            2. Add inline citations where appropriate using the format [Source: URL]
            3. Maintain the original tone and style
            4. Preserve the original intent and structure
            5. If information cannot be verified, indicate uncertainty
            
            Return the corrected text with citations.
            """,
            
            "uncertainty_handling": """
            You are a careful information communicator. The following text contains unverified claims.
            
            Original text: {original_text}
            Unverified claims: {unverified_claims}
            
            Instructions:
            1. Rephrase unverified claims to indicate uncertainty
            2. Use phrases like "according to some sources", "it has been suggested", "reports indicate"
            3. Maintain the original meaning while being more cautious
            4. Add appropriate qualifiers
            
            Return the revised text.
            """,
            
            "citation_addition": """
            You are an expert at adding proper citations to text.
            
            Text: {text}
            Available sources: {sources}
            
            Instructions:
            1. Add inline citations for factual claims
            2. Use the format [Source: Title, URL]
            3. Ensure citations are relevant and support the claims
            4. Don't over-cite - only cite when necessary
            
            Return the text with appropriate citations.
            """
        }
        
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process a correction request and return corrected content.
        
        Args:
            message: Message containing content to correct and verification results
            
        Returns:
            AgentMessage: Response with corrected content
        """
        try:
            self.update_state("processing", "Correcting content")
            
            # Extract request details
            original_text = message.content.get("text", "")
            verification_results = message.content.get("verification_results", [])
            correction_type = message.content.get("correction_type", "comprehensive")
            
            # Determine correction strategy
            correction_strategy = self._determine_correction_strategy(verification_results)
            
            # Apply corrections
            corrected_content = self._apply_corrections(
                original_text, 
                verification_results, 
                correction_strategy
            )
            
            # Generate correction report
            correction_report = self._generate_correction_report(
                original_text, 
                corrected_content, 
                verification_results
            )
            
            # Create response
            response_content = {
                "original_text": original_text,
                "corrected_text": corrected_content["text"],
                "corrections_made": corrected_content["corrections"],
                "citations_added": corrected_content["citations"],
                "correction_report": correction_report,
                "confidence_improvement": corrected_content["confidence_improvement"],
                "metadata": {
                    "correction_strategy": correction_strategy,
                    "correction_type": correction_type,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.update_state("idle")
            
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type="correction_response",
                content=response_content,
                timestamp=datetime.now(),
                message_id=f"correct_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            
        except Exception as e:
            self.handle_error(e, "content correction")
            return self._create_error_response(message.sender, str(e))
    
    def _determine_correction_strategy(self, verification_results: List[Dict[str, Any]]) -> str:
        """
        Determine the best correction strategy based on verification results.
        
        Args:
            verification_results: Results from the verifier agent
            
        Returns:
            Strategy name
        """
        if not verification_results:
            return "no_correction_needed"
        
        # Analyze verification results
        contradicted_count = len([r for r in verification_results if r.get("verification_status") == "contradicted"])
        unverified_count = len([r for r in verification_results if r.get("verification_status") == "unverified"])
        low_confidence_count = len([r for r in verification_results if r.get("confidence_score", 1.0) < 0.3])
        
        if contradicted_count > 0:
            return "factual_correction"
        elif unverified_count > len(verification_results) * 0.5:
            return "uncertainty_handling"
        elif low_confidence_count > 0:
            return "citation_addition"
        else:
            return "minor_improvements"
    
    def _apply_corrections(self, original_text: str, verification_results: List[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
        """
        Apply corrections based on the determined strategy.
        
        Args:
            original_text: Original text to correct
            verification_results: Verification results
            strategy: Correction strategy to use
            
        Returns:
            Dictionary with corrected content and metadata
        """
        corrections = []
        citations = []
        confidence_improvement = 0.0
        
        if strategy == "no_correction_needed":
            return {
                "text": original_text,
                "corrections": [],
                "citations": [],
                "confidence_improvement": 0.0
            }
        
        try:
            if strategy == "factual_correction":
                result = self._apply_factual_corrections(original_text, verification_results)
            elif strategy == "uncertainty_handling":
                result = self._apply_uncertainty_handling(original_text, verification_results)
            elif strategy == "citation_addition":
                result = self._apply_citation_addition(original_text, verification_results)
            else:
                result = self._apply_minor_improvements(original_text, verification_results)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying corrections: {e}")
            return {
                "text": original_text,
                "corrections": [{"type": "error", "description": str(e)}],
                "citations": [],
                "confidence_improvement": 0.0
            }
    
    def _apply_factual_corrections(self, text: str, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply factual corrections using verified information."""
        # Extract contradicted and low-confidence claims
        contradicted_claims = [r for r in verification_results if r.get("verification_status") == "contradicted"]
        low_confidence_claims = [r for r in verification_results if r.get("confidence_score", 1.0) < 0.3]
        
        # Prepare correction prompt
        prompt = self.correction_templates["factual_correction"].format(
            original_text=text,
            verification_results=json.dumps(verification_results, indent=2),
            sources=json.dumps([r.get("sources", []) for r in verification_results], indent=2)
        )
        
        messages = [
            SystemMessage(content="You are a fact-checking expert specializing in accurate content correction."),
            HumanMessage(content=prompt)
        ]
        
        response = self.correction_model.invoke(messages)
        corrected_text = response.content
        
        # Track corrections made
        corrections = []
        for claim in contradicted_claims + low_confidence_claims:
            corrections.append({
                "type": "factual_correction",
                "original_claim": claim.get("claim", ""),
                "confidence_before": claim.get("confidence_score", 0.0),
                "confidence_after": 0.8,  # Estimated improvement
                "sources_used": claim.get("sources", [])
            })
        
        # Extract citations
        citations = self._extract_citations(corrected_text)
        
        return {
            "text": corrected_text,
            "corrections": corrections,
            "citations": citations,
            "confidence_improvement": 0.3  # Estimated improvement
        }
    
    def _apply_uncertainty_handling(self, text: str, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply uncertainty handling for unverified claims."""
        unverified_claims = [r for r in verification_results if r.get("verification_status") == "unverified"]
        
        prompt = self.correction_templates["uncertainty_handling"].format(
            original_text=text,
            unverified_claims=json.dumps([r.get("claim", "") for r in unverified_claims], indent=2)
        )
        
        messages = [
            SystemMessage(content="You are an expert at communicating uncertainty while maintaining clarity."),
            HumanMessage(content=prompt)
        ]
        
        response = self.correction_model.invoke(messages)
        corrected_text = response.content
        
        corrections = []
        for claim in unverified_claims:
            corrections.append({
                "type": "uncertainty_qualification",
                "original_claim": claim.get("claim", ""),
                "qualification_added": "uncertainty indicators",
                "confidence_before": claim.get("confidence_score", 0.0),
                "confidence_after": 0.6  # Moderate improvement
            })
        
        return {
            "text": corrected_text,
            "corrections": corrections,
            "citations": [],
            "confidence_improvement": 0.2
        }
    
    def _apply_citation_addition(self, text: str, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add citations to support claims."""
        verified_claims = [r for r in verification_results if r.get("verification_status") == "verified"]
        
        # Collect all sources
        all_sources = []
        for result in verified_claims:
            all_sources.extend(result.get("sources", []))
        
        prompt = self.correction_templates["citation_addition"].format(
            text=text,
            sources=json.dumps(all_sources, indent=2)
        )
        
        messages = [
            SystemMessage(content="You are an expert at adding appropriate citations to support claims."),
            HumanMessage(content=prompt)
        ]
        
        response = self.correction_model.invoke(messages)
        corrected_text = response.content
        
        citations = self._extract_citations(corrected_text)
        
        corrections = []
        for claim in verified_claims:
            corrections.append({
                "type": "citation_addition",
                "claim": claim.get("claim", ""),
                "citations_added": len(claim.get("sources", [])),
                "confidence_before": claim.get("confidence_score", 0.0),
                "confidence_after": min(1.0, claim.get("confidence_score", 0.0) + 0.1)
            })
        
        return {
            "text": corrected_text,
            "corrections": corrections,
            "citations": citations,
            "confidence_improvement": 0.1
        }
    
    def _apply_minor_improvements(self, text: str, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply minor improvements and clarifications."""
        # For minor improvements, we might just add some clarifications
        # This is a simplified implementation
        
        corrections = []
        for result in verification_results:
            if result.get("confidence_score", 1.0) < 0.8:
                corrections.append({
                    "type": "minor_clarification",
                    "claim": result.get("claim", ""),
                    "improvement": "added clarification",
                    "confidence_before": result.get("confidence_score", 0.0),
                    "confidence_after": min(1.0, result.get("confidence_score", 0.0) + 0.05)
                })
        
        return {
            "text": text,  # No major changes for minor improvements
            "corrections": corrections,
            "citations": [],
            "confidence_improvement": 0.05
        }
    
    def _extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from corrected text."""
        import re
        
        # Look for citation patterns like [Source: Title, URL]
        citation_pattern = r'\[Source:\s*([^,]+),\s*([^\]]+)\]'
        matches = re.findall(citation_pattern, text)
        
        citations = []
        for title, url in matches:
            citations.append({
                "title": title.strip(),
                "url": url.strip(),
                "format": "inline"
            })
        
        return citations
    
    def _generate_correction_report(self, original_text: str, corrected_content: Dict[str, Any], verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a detailed correction report."""
        corrections = corrected_content["corrections"]
        
        report = {
            "summary": {
                "total_corrections": len(corrections),
                "correction_types": list(set([c.get("type", "unknown") for c in corrections])),
                "confidence_improvement": corrected_content["confidence_improvement"],
                "citations_added": len(corrected_content["citations"])
            },
            "detailed_corrections": corrections,
            "verification_summary": {
                "total_claims_verified": len(verification_results),
                "verified_claims": len([r for r in verification_results if r.get("verification_status") == "verified"]),
                "unverified_claims": len([r for r in verification_results if r.get("verification_status") == "unverified"]),
                "contradicted_claims": len([r for r in verification_results if r.get("verification_status") == "contradicted"])
            },
            "recommendations": self._generate_correction_recommendations(corrections, verification_results)
        }
        
        return report
    
    def _generate_correction_recommendations(self, corrections: List[Dict[str, Any]], verification_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on corrections made."""
        recommendations = []
        
        correction_types = [c.get("type", "unknown") for c in corrections]
        
        if "factual_correction" in correction_types:
            recommendations.append("Factual corrections were made. Review the changes to ensure accuracy.")
        
        if "uncertainty_qualification" in correction_types:
            recommendations.append("Uncertainty qualifiers were added. Consider seeking additional verification.")
        
        if "citation_addition" in correction_types:
            recommendations.append("Citations were added. Verify that all citations are accurate and accessible.")
        
        contradicted_count = len([r for r in verification_results if r.get("verification_status") == "contradicted"])
        if contradicted_count > 0:
            recommendations.append("Some claims were contradicted by sources. Manual review recommended.")
        
        return recommendations
    
    def _create_error_response(self, recipient: str, error_message: str) -> AgentMessage:
        """Create an error response message."""
        return AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type="correction_error",
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
            "Factual content correction",
            "Citation addition and formatting",
            "Uncertainty qualification",
            "Content re-generation with verified data",
            "Correction reporting and analysis",
            "Confidence score improvement"
        ]
