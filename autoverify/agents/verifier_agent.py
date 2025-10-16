"""
Verifier Agent - Fact Verification and Hallucination Detection
============================================================

The Verifier Agent is responsible for:
1. Parsing generated responses into claim units
2. Using RAG to search authoritative sources
3. Assigning credibility scores based on source reliability
4. Detecting potential hallucinations and inconsistencies
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Optional imports
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    VECTOR_STORES_AVAILABLE = True
except ImportError:
    VECTOR_STORES_AVAILABLE = False

from ..core.base_agent import BaseAgent, AgentMessage, VerificationResult

class VerifierAgent(BaseAgent):
    """
    Verifier Agent for fact-checking and hallucination detection.
    
    This agent handles:
    - Claim extraction and parsing
    - Source verification using RAG
    - Credibility scoring
    - Hallucination detection
    """
    
    def __init__(self, agent_id: str = "verifier", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "verifier")
        
        # Configuration
        self.config = config or {
            "confidence_threshold": 0.7,
            "max_sources": 5,
            "embedding_model": "text-embedding-3-small",
            "verification_model": "gpt-4"
        }
        
        # Initialize models
        self.embedding_model = OpenAIEmbeddings(
            model=self.config.get("embedding_model", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.verification_model = ChatOpenAI(
            model=self.config.get("verification_model", "gpt-4"), 
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Knowledge sources
        self.knowledge_sources = {
            "wikipedia": "https://en.wikipedia.org/wiki/",
            "pubmed": "https://pubmed.ncbi.nlm.nih.gov/",
            "arxiv": "https://arxiv.org/",
            "reuters": "https://www.reuters.com/",
            "bbc": "https://www.bbc.com/"
        }
        
        # Vector stores for different domains
        self.vector_stores = {}
        self._initialize_vector_stores()
        
        # Claim patterns for extraction
        self.claim_patterns = [
            r"([A-Z][^.!?]*\s+(?:is|are|was|were|has|have|had|will|would|can|could|should|must)\s+[^.!?]*[.!?])",
            r"([A-Z][^.!?]*\s+(?:according to|based on|studies show|research indicates)\s+[^.!?]*[.!?])",
            r"([A-Z][^.!?]*\s+(?:in|during|on|at)\s+\d{4}[^.!?]*[.!?])",
            r"([A-Z][^.!?]*\s+(?:percent|%|\$|dollars|years|months|days)[^.!?]*[.!?])"
        ]
        
    def _initialize_vector_stores(self) -> None:
        """Initialize vector stores for different knowledge domains."""
        try:
            # This would typically load from pre-built indexes
            # For now, we'll create empty stores
            self.vector_stores = {
                "general": None,
                "scientific": None,
                "financial": None,
                "medical": None
            }
            self.logger.info("Vector stores initialized")
        except Exception as e:
            self.logger.error(f"Error initializing vector stores: {e}")
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process a verification request and return verification results.
        
        Args:
            message: Message containing text to verify
            
        Returns:
            AgentMessage: Response with verification results
        """
        try:
            self.update_state("processing", "Verifying claims")
            
            # Extract content to verify
            text_to_verify = message.content.get("text", "")
            context = message.content.get("context", {})
            verification_level = message.content.get("level", "standard")
            
            # Extract claims from text
            claims = self._extract_claims(text_to_verify)
            
            # Verify each claim
            verification_results = []
            for claim in claims:
                result = self._verify_claim(claim, context, verification_level)
                verification_results.append(result)
            
            # Calculate overall verification score
            overall_score = self._calculate_overall_score(verification_results)
            
            # Detect potential hallucinations
            hallucination_analysis = self._analyze_hallucinations(verification_results)
            
            # Create response
            response_content = {
                "verification_results": [self._result_to_dict(r) for r in verification_results],
                "overall_confidence": overall_score,
                "hallucination_analysis": hallucination_analysis,
                "verification_metadata": {
                    "total_claims": len(claims),
                    "verified_claims": len([r for r in verification_results if r.verification_status == "verified"]),
                    "unverified_claims": len([r for r in verification_results if r.verification_status == "unverified"]),
                    "contradicted_claims": len([r for r in verification_results if r.verification_status == "contradicted"]),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.update_state("idle")
            
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type="verification_response",
                content=response_content,
                timestamp=datetime.now(),
                message_id=f"verify_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            
        except Exception as e:
            self.handle_error(e, "claim verification")
            return self._create_error_response(message.sender, str(e))
    
    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text using pattern matching and NLP.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of extracted claims
        """
        claims = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Check against claim patterns
            for pattern in self.claim_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                claims.extend(matches)
        
        # Remove duplicates and clean up
        claims = list(set([claim.strip() for claim in claims if claim.strip()]))
        
        # Use LLM to refine claim extraction
        refined_claims = self._refine_claims_with_llm(text, claims)
        
        return refined_claims
    
    def _refine_claims_with_llm(self, text: str, initial_claims: List[str]) -> List[str]:
        """
        Use LLM to refine and improve claim extraction.
        
        Args:
            text: Original text
            initial_claims: Initially extracted claims
            
        Returns:
            Refined list of claims
        """
        try:
            prompt = f"""
            Extract all factual claims from the following text. Focus on statements that can be verified or contradicted.
            
            Text: {text}
            
            Initial claims found: {initial_claims}
            
            Return a JSON list of refined claims. Each claim should be:
            1. A complete, verifiable statement
            2. Factual rather than opinion
            3. Specific enough to be checked against sources
            
            Format: ["claim1", "claim2", ...]
            """
            
            messages = [
                SystemMessage(content="You are an expert at extracting factual claims from text."),
                HumanMessage(content=prompt)
            ]
            
            response = self.verification_model.invoke(messages)
            
            # Parse JSON response
            try:
                refined_claims = json.loads(response.content)
                return refined_claims if isinstance(refined_claims, list) else initial_claims
            except json.JSONDecodeError:
                return initial_claims
                
        except Exception as e:
            self.logger.warning(f"Error refining claims with LLM: {e}")
            return initial_claims
    
    def _verify_claim(self, claim: str, context: Dict[str, Any], level: str) -> VerificationResult:
        """
        Verify a single claim against available sources.
        
        Args:
            claim: The claim to verify
            context: Additional context
            level: Verification level (basic, standard, thorough)
            
        Returns:
            VerificationResult object
        """
        try:
            # Search for relevant sources
            sources = self._search_sources(claim, level)
            
            # Analyze source reliability
            source_analysis = self._analyze_source_reliability(sources)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(claim, sources, source_analysis)
            
            # Determine verification status
            verification_status = self._determine_verification_status(confidence_score)
            
            # Extract evidence
            evidence = self._extract_evidence(sources, claim)
            
            return VerificationResult(
                claim=claim,
                confidence_score=confidence_score,
                sources=[s["url"] for s in sources],
                verification_status=verification_status,
                evidence=evidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error verifying claim '{claim}': {e}")
            return VerificationResult(
                claim=claim,
                confidence_score=0.0,
                sources=[],
                verification_status="error",
                evidence=[],
                timestamp=datetime.now()
            )
    
    def _search_sources(self, claim: str, level: str) -> List[Dict[str, Any]]:
        """
        Search for sources that can verify the claim.
        
        Args:
            claim: The claim to search for
            level: Search level (basic, standard, thorough)
            
        Returns:
            List of source documents with metadata
        """
        sources = []
        
        try:
            # Generate search queries
            search_queries = self._generate_search_queries(claim)
            
            # Search vector stores
            for query in search_queries[:3]:  # Limit to top 3 queries
                # This would typically search against pre-built vector stores
                # For now, we'll simulate the search
                mock_sources = self._mock_source_search(query, level)
                sources.extend(mock_sources)
            
            # Remove duplicates and rank by relevance
            sources = self._deduplicate_and_rank_sources(sources)
            
            return sources[:self.config["max_sources"]]
            
        except Exception as e:
            self.logger.error(f"Error searching sources: {e}")
            return []
    
    def _generate_search_queries(self, claim: str) -> List[str]:
        """Generate search queries from a claim."""
        # Simple query generation - in practice, this would be more sophisticated
        words = claim.split()
        queries = [
            claim,  # Original claim
            " ".join(words[:5]),  # First 5 words
            " ".join(words[-5:]) if len(words) > 5 else claim,  # Last 5 words
        ]
        
        # Add domain-specific queries
        if any(word in claim.lower() for word in ["study", "research", "scientists"]):
            queries.append(f"research study {claim}")
        
        return queries
    
    def _mock_source_search(self, query: str, level: str) -> List[Dict[str, Any]]:
        """
        Mock source search - in production, this would search real vector stores.
        
        Args:
            query: Search query
            level: Search level
            
        Returns:
            List of mock source documents
        """
        # This is a mock implementation
        # In production, you would search against real vector stores
        mock_sources = [
            {
                "url": "https://example.com/source1",
                "title": f"Source for: {query}",
                "content": f"Relevant information about {query}",
                "reliability_score": 0.8,
                "domain": "example.com"
            },
            {
                "url": "https://example.com/source2", 
                "title": f"Additional info: {query}",
                "content": f"More details about {query}",
                "reliability_score": 0.7,
                "domain": "example.com"
            }
        ]
        
        return mock_sources
    
    def _deduplicate_and_rank_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and rank sources by relevance."""
        # Remove duplicates by URL
        seen_urls = set()
        unique_sources = []
        for source in sources:
            if source["url"] not in seen_urls:
                seen_urls.add(source["url"])
                unique_sources.append(source)
        
        # Sort by reliability score
        unique_sources.sort(key=lambda x: x["reliability_score"], reverse=True)
        
        return unique_sources
    
    def _analyze_source_reliability(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the reliability of sources."""
        if not sources:
            return {"average_reliability": 0.0, "reliable_sources": 0, "total_sources": 0}
        
        reliability_scores = [s["reliability_score"] for s in sources]
        average_reliability = sum(reliability_scores) / len(reliability_scores)
        reliable_sources = len([s for s in sources if s["reliability_score"] > 0.7])
        
        return {
            "average_reliability": average_reliability,
            "reliable_sources": reliable_sources,
            "total_sources": len(sources),
            "reliability_distribution": {
                "high": len([s for s in sources if s["reliability_score"] > 0.8]),
                "medium": len([s for s in sources if 0.5 <= s["reliability_score"] <= 0.8]),
                "low": len([s for s in sources if s["reliability_score"] < 0.5])
            }
        }
    
    def _calculate_confidence_score(self, claim: str, sources: List[Dict[str, Any]], source_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for a claim."""
        if not sources:
            return 0.0
        
        # Base score from source reliability
        base_score = source_analysis["average_reliability"]
        
        # Adjust based on number of sources
        source_count_factor = min(1.0, len(sources) / 3.0)
        
        # Adjust based on claim specificity (simplified)
        specificity_factor = 0.8  # Placeholder - would analyze claim specificity
        
        # Calculate final score
        confidence_score = base_score * source_count_factor * specificity_factor
        
        return min(1.0, max(0.0, confidence_score))
    
    def _determine_verification_status(self, confidence_score: float) -> str:
        """Determine verification status based on confidence score."""
        if confidence_score >= self.config["confidence_threshold"]:
            return "verified"
        elif confidence_score >= 0.3:
            return "unverified"
        else:
            return "contradicted"
    
    def _extract_evidence(self, sources: List[Dict[str, Any]], claim: str) -> List[Dict[str, Any]]:
        """Extract supporting evidence from sources."""
        evidence = []
        
        for source in sources:
            evidence.append({
                "source_url": source["url"],
                "source_title": source["title"],
                "relevant_content": source["content"][:200] + "..." if len(source["content"]) > 200 else source["content"],
                "reliability_score": source["reliability_score"],
                "relevance_score": 0.8  # Placeholder - would calculate actual relevance
            })
        
        return evidence
    
    def _calculate_overall_score(self, results: List[VerificationResult]) -> float:
        """Calculate overall verification score."""
        if not results:
            return 0.0
        
        scores = [r.confidence_score for r in results]
        return sum(scores) / len(scores)
    
    def _analyze_hallucinations(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """Analyze potential hallucinations in the results."""
        total_claims = len(results)
        verified_claims = len([r for r in results if r.verification_status == "verified"])
        unverified_claims = len([r for r in results if r.verification_status == "unverified"])
        contradicted_claims = len([r for r in results if r.verification_status == "contradicted"])
        
        hallucination_risk = "low"
        if contradicted_claims > 0:
            hallucination_risk = "high"
        elif unverified_claims > total_claims * 0.5:
            hallucination_risk = "medium"
        
        return {
            "hallucination_risk": hallucination_risk,
            "risk_factors": {
                "contradicted_claims": contradicted_claims,
                "unverified_claims": unverified_claims,
                "low_confidence_claims": len([r for r in results if r.confidence_score < 0.3])
            },
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: List[VerificationResult]) -> List[str]:
        """Generate recommendations based on verification results."""
        recommendations = []
        
        contradicted_count = len([r for r in results if r.verification_status == "contradicted"])
        unverified_count = len([r for r in results if r.verification_status == "unverified"])
        
        if contradicted_count > 0:
            recommendations.append("High risk of hallucination detected. Manual review recommended.")
        
        if unverified_count > len(results) * 0.3:
            recommendations.append("Many claims could not be verified. Additional fact-checking needed.")
        
        if all(r.confidence_score < 0.5 for r in results):
            recommendations.append("Overall confidence is low. Consider using more reliable sources.")
        
        return recommendations
    
    def _result_to_dict(self, result: VerificationResult) -> Dict[str, Any]:
        """Convert VerificationResult to dictionary."""
        return {
            "claim": result.claim,
            "confidence_score": result.confidence_score,
            "sources": result.sources,
            "verification_status": result.verification_status,
            "evidence": result.evidence,
            "timestamp": result.timestamp.isoformat()
        }
    
    def _create_error_response(self, recipient: str, error_message: str) -> AgentMessage:
        """Create an error response message."""
        return AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type="verification_error",
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
            "Factual claim extraction",
            "Source verification using RAG",
            "Credibility scoring",
            "Hallucination detection",
            "Evidence extraction",
            "Multi-source cross-referencing"
        ]
