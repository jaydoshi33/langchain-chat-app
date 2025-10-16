"""
Audit & Feedback Agent - Logging and Explainable Verification Reports
==================================================================

The Audit & Feedback Agent is responsible for:
1. Logging all verification steps to structured database
2. Providing explainable "Verification Reports" for enterprise use
3. Tracking system performance and accuracy metrics
4. Generating insights and recommendations for improvement
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.base_agent import BaseAgent, AgentMessage

class AuditAgent(BaseAgent):
    """
    Audit & Feedback Agent for logging and reporting.
    
    This agent handles:
    - Structured logging of all verification activities
    - Generation of explainable verification reports
    - Performance tracking and analytics
    - Enterprise reporting and compliance
    """
    
    def __init__(self, agent_id: str = "auditor", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "auditor")
        
        # Configuration
        self.config = config or {
            "database_path": "autoverify_audit.db",
            "report_model": "gpt-4",
            "retention_days": 90,
            "report_formats": ["json", "html", "pdf"]
        }
        
        # Initialize report generation model
        self.report_model = ChatOpenAI(
            model=self.config["report_model"],
            temperature=0.1  # Low temperature for consistent reporting
        )
        
        # Initialize database
        self.db_path = Path(self.config["database_path"])
        self._initialize_database()
        
        # Report templates
        self.report_templates = {
            "executive_summary": """
            Generate an executive summary for a verification report.
            
            Verification Session: {session_id}
            Total Claims: {total_claims}
            Verified Claims: {verified_claims}
            Unverified Claims: {unverified_claims}
            Contradicted Claims: {contradicted_claims}
            Overall Confidence: {overall_confidence}
            Hallucination Risk: {hallucination_risk}
            
            Provide a concise executive summary suitable for business stakeholders.
            """,
            
            "technical_details": """
            Generate technical details for a verification report.
            
            Verification Results: {verification_results}
            Correction Details: {correction_details}
            Source Analysis: {source_analysis}
            Performance Metrics: {performance_metrics}
            
            Provide detailed technical information for technical stakeholders.
            """,
            
            "compliance_report": """
            Generate a compliance-focused verification report.
            
            Session Details: {session_details}
            Verification Process: {verification_process}
            Source Reliability: {source_reliability}
            Audit Trail: {audit_trail}
            
            Focus on compliance, traceability, and regulatory requirements.
            """
        }
        
    def _initialize_database(self) -> None:
        """Initialize the audit database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS verification_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        query TEXT,
                        timestamp DATETIME,
                        overall_confidence REAL,
                        hallucination_risk TEXT,
                        processing_time REAL,
                        model_used TEXT
                    )
                """)
                
                # Create claims table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS verification_claims (
                        claim_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        claim_text TEXT,
                        confidence_score REAL,
                        verification_status TEXT,
                        sources_count INTEGER,
                        evidence_count INTEGER,
                        timestamp DATETIME,
                        FOREIGN KEY (session_id) REFERENCES verification_sessions (session_id)
                    )
                """)
                
                # Create sources table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS verification_sources (
                        source_id TEXT PRIMARY KEY,
                        claim_id TEXT,
                        url TEXT,
                        title TEXT,
                        reliability_score REAL,
                        domain TEXT,
                        timestamp DATETIME,
                        FOREIGN KEY (claim_id) REFERENCES verification_claims (claim_id)
                    )
                """)
                
                # Create corrections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS corrections (
                        correction_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        original_text TEXT,
                        corrected_text TEXT,
                        correction_type TEXT,
                        confidence_improvement REAL,
                        citations_added INTEGER,
                        timestamp DATETIME,
                        FOREIGN KEY (session_id) REFERENCES verification_sessions (session_id)
                    )
                """)
                
                # Create performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        metric_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        metric_name TEXT,
                        metric_value REAL,
                        timestamp DATETIME,
                        FOREIGN KEY (session_id) REFERENCES verification_sessions (session_id)
                    )
                """)
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process an audit request and return audit report.
        
        Args:
            message: Message containing verification data to audit
            
        Returns:
            AgentMessage: Response with audit report
        """
        try:
            self.update_state("processing", "Generating audit report")
            
            # Extract audit data
            session_data = message.content.get("session_data", {})
            verification_results = message.content.get("verification_results", [])
            correction_results = message.content.get("correction_results", {})
            report_type = message.content.get("report_type", "comprehensive")
            
            # Log the session data
            session_id = self._log_verification_session(session_data, verification_results, correction_results)
            
            # Generate audit report
            audit_report = self._generate_audit_report(
                session_id, 
                verification_results, 
                correction_results, 
                report_type
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(session_id)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(verification_results, correction_results)
            
            # Create response
            response_content = {
                "session_id": session_id,
                "audit_report": audit_report,
                "performance_metrics": performance_metrics,
                "recommendations": recommendations,
                "compliance_status": self._assess_compliance_status(verification_results),
                "metadata": {
                    "report_type": report_type,
                    "generated_at": datetime.now().isoformat(),
                    "data_retention_until": (datetime.now() + timedelta(days=self.config["retention_days"])).isoformat()
                }
            }
            
            self.update_state("idle")
            
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type="audit_response",
                content=response_content,
                timestamp=datetime.now(),
                message_id=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            
        except Exception as e:
            self.handle_error(e, "audit report generation")
            return self._create_error_response(message.sender, str(e))
    
    def _log_verification_session(self, session_data: Dict[str, Any], verification_results: List[Dict[str, Any]], correction_results: Dict[str, Any]) -> str:
        """Log verification session data to database."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Log session
                cursor.execute("""
                    INSERT INTO verification_sessions 
                    (session_id, user_id, query, timestamp, overall_confidence, hallucination_risk, processing_time, model_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    session_data.get("user_id", "anonymous"),
                    session_data.get("query", ""),
                    datetime.now().isoformat(),
                    session_data.get("overall_confidence", 0.0),
                    session_data.get("hallucination_risk", "unknown"),
                    session_data.get("processing_time", 0.0),
                    session_data.get("model_used", "unknown")
                ))
                
                # Log claims
                for i, result in enumerate(verification_results):
                    claim_id = f"{session_id}_claim_{i}"
                    cursor.execute("""
                        INSERT INTO verification_claims 
                        (claim_id, session_id, claim_text, confidence_score, verification_status, sources_count, evidence_count, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        claim_id,
                        session_id,
                        result.get("claim", ""),
                        result.get("confidence_score", 0.0),
                        result.get("verification_status", "unknown"),
                        len(result.get("sources", [])),
                        len(result.get("evidence", [])),
                        datetime.now().isoformat()
                    ))
                    
                    # Log sources
                    for j, source in enumerate(result.get("sources", [])):
                        source_id = f"{claim_id}_source_{j}"
                        cursor.execute("""
                            INSERT INTO verification_sources 
                            (source_id, claim_id, url, title, reliability_score, domain, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            source_id,
                            claim_id,
                            source,
                            f"Source {j+1}",
                            0.8,  # Default reliability score
                            self._extract_domain(source),
                            datetime.now().isoformat()
                        ))
                
                # Log corrections
                if correction_results:
                    cursor.execute("""
                        INSERT INTO corrections 
                        (correction_id, session_id, original_text, corrected_text, correction_type, confidence_improvement, citations_added, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"{session_id}_correction",
                        session_id,
                        correction_results.get("original_text", ""),
                        correction_results.get("corrected_text", ""),
                        correction_results.get("correction_type", "unknown"),
                        correction_results.get("confidence_improvement", 0.0),
                        len(correction_results.get("citations_added", [])),
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                self.logger.info(f"Logged verification session: {session_id}")
                
        except Exception as e:
            self.logger.error(f"Error logging session: {e}")
        
        return session_id
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"
    
    def _generate_audit_report(self, session_id: str, verification_results: List[Dict[str, Any]], correction_results: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        try:
            # Calculate summary statistics
            total_claims = len(verification_results)
            verified_claims = len([r for r in verification_results if r.get("verification_status") == "verified"])
            unverified_claims = len([r for r in verification_results if r.get("verification_status") == "unverified"])
            contradicted_claims = len([r for r in verification_results if r.get("verification_status") == "contradicted"])
            
            overall_confidence = sum(r.get("confidence_score", 0.0) for r in verification_results) / max(total_claims, 1)
            
            # Determine hallucination risk
            hallucination_risk = "low"
            if contradicted_claims > 0:
                hallucination_risk = "high"
            elif unverified_claims > total_claims * 0.5:
                hallucination_risk = "medium"
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                session_id, total_claims, verified_claims, unverified_claims, 
                contradicted_claims, overall_confidence, hallucination_risk
            )
            
            # Generate technical details
            technical_details = self._generate_technical_details(verification_results, correction_results)
            
            # Generate compliance report
            compliance_report = self._generate_compliance_report(session_id, verification_results)
            
            return {
                "executive_summary": executive_summary,
                "technical_details": technical_details,
                "compliance_report": compliance_report,
                "summary_statistics": {
                    "total_claims": total_claims,
                    "verified_claims": verified_claims,
                    "unverified_claims": unverified_claims,
                    "contradicted_claims": contradicted_claims,
                    "overall_confidence": overall_confidence,
                    "hallucination_risk": hallucination_risk
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating audit report: {e}")
            return {"error": str(e)}
    
    def _generate_executive_summary(self, session_id: str, total_claims: int, verified_claims: int, 
                                  unverified_claims: int, contradicted_claims: int, 
                                  overall_confidence: float, hallucination_risk: str) -> str:
        """Generate executive summary using LLM."""
        try:
            prompt = self.report_templates["executive_summary"].format(
                session_id=session_id,
                total_claims=total_claims,
                verified_claims=verified_claims,
                unverified_claims=unverified_claims,
                contradicted_claims=contradicted_claims,
                overall_confidence=overall_confidence,
                hallucination_risk=hallucination_risk
            )
            
            messages = [
                SystemMessage(content="You are an expert at creating executive summaries for technical reports."),
                HumanMessage(content=prompt)
            ]
            
            response = self.report_model.invoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return f"Executive summary generation failed: {str(e)}"
    
    def _generate_technical_details(self, verification_results: List[Dict[str, Any]], correction_results: Dict[str, Any]) -> str:
        """Generate technical details using LLM."""
        try:
            prompt = self.report_templates["technical_details"].format(
                verification_results=json.dumps(verification_results, indent=2),
                correction_details=json.dumps(correction_results, indent=2),
                source_analysis=self._analyze_sources(verification_results),
                performance_metrics=self._get_performance_metrics()
            )
            
            messages = [
                SystemMessage(content="You are a technical expert specializing in AI verification systems."),
                HumanMessage(content=prompt)
            ]
            
            response = self.report_model.invoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating technical details: {e}")
            return f"Technical details generation failed: {str(e)}"
    
    def _generate_compliance_report(self, session_id: str, verification_results: List[Dict[str, Any]]) -> str:
        """Generate compliance report using LLM."""
        try:
            prompt = self.report_templates["compliance_report"].format(
                session_details=f"Session ID: {session_id}",
                verification_process="Multi-agent verification with RAG",
                source_reliability=self._assess_source_reliability(verification_results),
                audit_trail="Complete audit trail maintained in database"
            )
            
            messages = [
                SystemMessage(content="You are a compliance expert specializing in AI system auditing."),
                HumanMessage(content=prompt)
            ]
            
            response = self.report_model.invoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            return f"Compliance report generation failed: {str(e)}"
    
    def _analyze_sources(self, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze source reliability and distribution."""
        all_sources = []
        for result in verification_results:
            all_sources.extend(result.get("sources", []))
        
        if not all_sources:
            return {"total_sources": 0, "reliability_distribution": {}}
        
        # Analyze domains
        domains = {}
        for source in all_sources:
            domain = self._extract_domain(source)
            domains[domain] = domains.get(domain, 0) + 1
        
        return {
            "total_sources": len(all_sources),
            "unique_sources": len(set(all_sources)),
            "domain_distribution": domains,
            "average_sources_per_claim": len(all_sources) / max(len(verification_results), 1)
        }
    
    def _assess_source_reliability(self, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall source reliability."""
        all_sources = []
        for result in verification_results:
            all_sources.extend(result.get("sources", []))
        
        if not all_sources:
            return {"reliability_score": 0.0, "assessment": "No sources available"}
        
        # Simple reliability assessment based on domains
        reliable_domains = ["wikipedia.org", "pubmed.ncbi.nlm.nih.gov", "reuters.com", "bbc.com"]
        reliable_count = sum(1 for source in all_sources if any(domain in source for domain in reliable_domains))
        
        reliability_score = reliable_count / len(all_sources)
        
        assessment = "high" if reliability_score > 0.7 else "medium" if reliability_score > 0.4 else "low"
        
        return {
            "reliability_score": reliability_score,
            "assessment": assessment,
            "reliable_sources": reliable_count,
            "total_sources": len(all_sources)
        }
    
    def _calculate_performance_metrics(self, session_id: str) -> Dict[str, Any]:
        """Calculate performance metrics for the session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get session data
                cursor.execute("SELECT * FROM verification_sessions WHERE session_id = ?", (session_id,))
                session_data = cursor.fetchone()
                
                if not session_data:
                    return {"error": "Session not found"}
                
                # Calculate metrics
                metrics = {
                    "processing_time": session_data[6],  # processing_time column
                    "overall_confidence": session_data[4],  # overall_confidence column
                    "total_claims": 0,
                    "verification_rate": 0.0,
                    "correction_rate": 0.0
                }
                
                # Get claims data
                cursor.execute("SELECT COUNT(*) FROM verification_claims WHERE session_id = ?", (session_id,))
                total_claims = cursor.fetchone()[0]
                metrics["total_claims"] = total_claims
                
                # Get verification rate
                cursor.execute("SELECT COUNT(*) FROM verification_claims WHERE session_id = ? AND verification_status = 'verified'", (session_id,))
                verified_claims = cursor.fetchone()[0]
                metrics["verification_rate"] = verified_claims / max(total_claims, 1)
                
                # Get correction rate
                cursor.execute("SELECT COUNT(*) FROM corrections WHERE session_id = ?", (session_id,))
                corrections_count = cursor.fetchone()[0]
                metrics["correction_rate"] = corrections_count / max(total_claims, 1)
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get overall statistics
                cursor.execute("SELECT COUNT(*) FROM verification_sessions")
                total_sessions = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(overall_confidence) FROM verification_sessions")
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                cursor.execute("SELECT AVG(processing_time) FROM verification_sessions")
                avg_processing_time = cursor.fetchone()[0] or 0.0
                
                return {
                    "total_sessions": total_sessions,
                    "average_confidence": avg_confidence,
                    "average_processing_time": avg_processing_time
                }
                
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, verification_results: List[Dict[str, Any]], correction_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on verification and correction results."""
        recommendations = []
        
        # Analyze verification results
        total_claims = len(verification_results)
        contradicted_claims = len([r for r in verification_results if r.get("verification_status") == "contradicted"])
        unverified_claims = len([r for r in verification_results if r.get("verification_status") == "unverified"])
        
        if contradicted_claims > 0:
            recommendations.append("High risk of hallucination detected. Consider implementing stricter fact-checking protocols.")
        
        if unverified_claims > total_claims * 0.3:
            recommendations.append("Many claims could not be verified. Consider expanding the knowledge base or improving source coverage.")
        
        # Analyze correction results
        if correction_results.get("confidence_improvement", 0) > 0.2:
            recommendations.append("Significant improvements were made through correction. The correction system is working effectively.")
        
        if len(correction_results.get("citations_added", [])) > 0:
            recommendations.append("Citations were added to improve credibility. Consider making citation addition automatic for all responses.")
        
        # General recommendations
        recommendations.extend([
            "Regular monitoring of verification accuracy is recommended.",
            "Consider implementing user feedback mechanisms to improve verification quality.",
            "Periodic review of source reliability scores is recommended."
        ])
        
        return recommendations
    
    def _assess_compliance_status(self, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess compliance status for enterprise use."""
        total_claims = len(verification_results)
        verified_claims = len([r for r in verification_results if r.get("verification_status") == "verified"])
        contradicted_claims = len([r for r in verification_results if r.get("verification_status") == "contradicted"])
        
        verification_rate = verified_claims / max(total_claims, 1)
        
        compliance_status = "compliant"
        if contradicted_claims > 0:
            compliance_status = "non_compliant"
        elif verification_rate < 0.7:
            compliance_status = "requires_review"
        
        return {
            "status": compliance_status,
            "verification_rate": verification_rate,
            "contradicted_claims": contradicted_claims,
            "audit_trail_complete": True,
            "source_traceability": True
        }
    
    def _create_error_response(self, recipient: str, error_message: str) -> AgentMessage:
        """Create an error response message."""
        return AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type="audit_error",
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
            "Structured audit logging",
            "Executive report generation",
            "Technical documentation",
            "Compliance reporting",
            "Performance analytics",
            "Recommendation generation"
        ]
    
    def get_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get audit summary for the last N days."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get session statistics
                cursor.execute("""
                    SELECT COUNT(*), AVG(overall_confidence), AVG(processing_time)
                    FROM verification_sessions 
                    WHERE timestamp >= ?
                """, (cutoff_date,))
                
                session_stats = cursor.fetchone()
                
                # Get claim statistics
                cursor.execute("""
                    SELECT verification_status, COUNT(*)
                    FROM verification_claims vc
                    JOIN verification_sessions vs ON vc.session_id = vs.session_id
                    WHERE vs.timestamp >= ?
                    GROUP BY verification_status
                """, (cutoff_date,))
                
                claim_stats = dict(cursor.fetchall())
                
                return {
                    "period_days": days,
                    "total_sessions": session_stats[0] or 0,
                    "average_confidence": session_stats[1] or 0.0,
                    "average_processing_time": session_stats[2] or 0.0,
                    "claim_verification_distribution": claim_stats
                }
                
        except Exception as e:
            self.logger.error(f"Error getting audit summary: {e}")
            return {"error": str(e)}
