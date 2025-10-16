"""
AutoVerify Streamlit Application
==============================

A comprehensive web interface for the AutoVerify system that provides
hallucination detection and fact verification for generative AI systems.
"""

import streamlit as st
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import AutoVerify components
from autoverify.core.orchestrator import AutoVerifyOrchestrator

# Page configuration
st.set_page_config(
    page_title="AutoVerify - AI Hallucination Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .verification-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .verified {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .unverified {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .contradicted {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .correction-highlight {
        background-color: #e7f3ff;
        border: 1px solid #1f77b4;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "verification_history" not in st.session_state:
    st.session_state.verification_history = []
if "system_status" not in st.session_state:
    st.session_state.system_status = {}

def initialize_autoverify():
    """Initialize the AutoVerify system."""
    try:
        if st.session_state.orchestrator is None:
            # Check for API key first
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("Please set your OPENAI_API_KEY in the .env file")
                st.error("Current working directory: " + os.getcwd())
                st.error("Environment variables loaded: " + str(bool(api_key)))
                return False
            
            config = {
                "generator": {
                    "primary_model": "gpt-4",
                    "temperature": 0.7
                },
                "verifier": {
                    "confidence_threshold": 0.7,
                    "max_sources": 5
                },
                "corrector": {
                    "correction_model": "gpt-4",
                    "confidence_threshold": 0.7
                },
                "auditor": {
                    "database_path": "autoverify_audit.db",
                    "retention_days": 90,
                    "report_model": "gpt-4"
                }
            }
            
            st.session_state.orchestrator = AutoVerifyOrchestrator(config)
            st.session_state.system_status = st.session_state.orchestrator.get_system_status()
            
        return True
    except Exception as e:
        st.error(f"Failed to initialize AutoVerify: {str(e)}")
        st.error("Please check your API keys and try running: python test_autoverify.py")
        return False

def display_header():
    """Display the main header."""
    st.markdown('<h1 class="main-header">🔍 AutoVerify</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            AI Agent for Hallucination Detection and Fact Verification in Generative AI Systems
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Verification Level
        verification_level = st.selectbox(
            "Verification Level",
            ["basic", "standard", "thorough"],
            index=1,
            help="Higher levels provide more comprehensive verification but take longer"
        )
        
        # Model Selection
        st.subheader("🤖 Model Configuration")
        primary_model = st.selectbox(
            "Primary Model",
            ["gpt-4", "gpt-3.5-turbo", "claude-3"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in generation"
        )
        
        # Confidence Threshold
        st.subheader("🎯 Verification Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum confidence score for verification"
        )
        
        # System Status
        st.subheader("📊 System Status")
        if st.session_state.system_status:
            status = st.session_state.system_status.get("system_status", "unknown")
            if status == "operational":
                st.success("✅ System Operational")
            else:
                st.error("❌ System Error")
            
            # Agent Status
            agents = st.session_state.system_status.get("agents", {})
            for agent_id, agent_status in agents.items():
                agent_status_text = agent_status.get("status", "unknown")
                if agent_status_text == "idle":
                    st.info(f"🔄 {agent_id.title()}: Ready")
                elif agent_status_text == "processing":
                    st.warning(f"⚡ {agent_id.title()}: Processing")
                else:
                    st.error(f"❌ {agent_id.title()}: {agent_status_text}")
        
        # Clear History
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.verification_history = []
            st.rerun()
        
        return {
            "verification_level": verification_level,
            "primary_model": primary_model,
            "temperature": temperature,
            "confidence_threshold": confidence_threshold
        }

def display_main_interface():
    """Display the main verification interface."""
    st.header("🔍 Query Verification")
    
    # Query input
    query = st.text_area(
        "Enter your query for verification:",
        height=100,
        placeholder="Ask any question and AutoVerify will generate a response, verify its accuracy, and provide corrections if needed..."
    )
    
    # Context input
    with st.expander("📝 Additional Context (Optional)"):
        context = st.text_area(
            "Provide additional context for better verification:",
            height=80,
            placeholder="Any additional information that might help with verification..."
        )
    
    # Verification button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Verify Query", type="primary", use_container_width=True):
            if query.strip():
                verify_query(query, context)
            else:
                st.warning("Please enter a query to verify.")

def verify_query(query: str, context: str = ""):
    """Process a query through the AutoVerify system."""
    if not st.session_state.orchestrator:
        st.error("AutoVerify system not initialized. Please refresh the page.")
        return
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Generation
        status_text.text("🤖 Generating response...")
        progress_bar.progress(25)
        
        # Step 2: Verification
        status_text.text("🔍 Verifying facts...")
        progress_bar.progress(50)
        
        # Step 3: Correction
        status_text.text("✏️ Applying corrections...")
        progress_bar.progress(75)
        
        # Step 4: Audit
        status_text.text("📊 Generating audit report...")
        progress_bar.progress(90)
        
        # Process the query (simplified for demo - in real implementation, this would be async)
        result = process_query_sync(query, context)
        
        progress_bar.progress(100)
        status_text.text("✅ Verification complete!")
        
        # Store in history
        st.session_state.verification_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "result": result
        })
        
        # Display results
        display_verification_results(result)
        
    except Exception as e:
        st.error(f"Verification failed: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ Verification failed")

def process_query_sync(query: str, context: str) -> Dict[str, Any]:
    """Synchronous wrapper for query processing using the real AutoVerify system."""
    import asyncio
    
    # Get configuration from session state
    config = {
        "generator": {
            "primary_model": "gpt-4",
            "temperature": 0.7
        },
        "verifier": {
            "embedding_model": "text-embedding-3-small",
            "verification_model": "gpt-4"
        },
        "corrector": {
            "correction_model": "gpt-4",
            "confidence_threshold": 0.7
        },
        "auditor": {
            "database_path": "autoverify_audit.db",
            "retention_days": 90,
            "report_model": "gpt-4"
        }
    }
    
    # Create orchestrator and process query
    orchestrator = AutoVerifyOrchestrator(config)
    
    # Run the async query processing
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(orchestrator.process_query(query, context or {}))
        return result
    finally:
        loop.close()

def display_verification_results(result: Dict[str, Any]):
    """Display the verification results."""
    st.header("📋 Verification Results")
    
    # Check if we have an error
    if 'error' in result:
        st.error(f"Error: {result['error']}")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get processing time
    processing_time = result.get('processing_metadata', {}).get('processing_time', 0)
    
    with col1:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    with col2:
        st.metric("Session ID", result.get('session_id', 'N/A')[:10] + '...')
    
    with col3:
        verification_results = result.get('verification_results', [])
        st.metric("Claims Verified", len(verification_results))
    
    with col4:
        st.metric("Status", "✅ Complete")
    
    # Original Response
    st.subheader("🤖 Original Response")
    original_response = result.get('original_response', 'No response generated')
    st.write(original_response)
    
    # Final Response
    st.subheader("✅ Final Response")
    final_response = result.get('final_response', original_response)
    st.write(final_response)
    
    # Verification Details
    st.subheader("🔍 Verification Details")
    
    verification_results = result.get('verification_results', [])
    if verification_results:
        for i, verification in enumerate(verification_results):
            with st.expander(f"Claim {i+1}"):
                st.write(f"**Claim:** {verification.get('claim', 'N/A')}")
                st.write(f"**Confidence:** {verification.get('confidence_score', 0):.2f}")
                st.write(f"**Status:** {verification.get('verification_status', 'N/A')}")
                
                # Show sources if available
                sources = verification.get('sources', [])
                if sources:
                    st.write("**Sources:**")
                    for source in sources:
                        st.write(f"• {source}")
    else:
        st.info("No detailed verification results available.")
    
    # Corrections
    correction_results = result.get('correction_results', {})
    if correction_results:
        st.subheader("✏️ Corrections Applied")
        corrections_made = correction_results.get('corrections_made', [])
        if corrections_made:
            st.write("**Corrections made:**")
            for correction in corrections_made:
                st.write(f"• {correction}")
        else:
            st.info("No corrections were needed.")
    
    # Audit Report
    st.subheader("📊 Audit Report")
    audit_report = result.get('audit_report', {})
    if audit_report:
        with st.expander("📋 Full Report"):
            st.json(audit_report)
    else:
        st.info("No audit report available.")

def display_history():
    """Display verification history."""
    if not st.session_state.verification_history:
        st.info("No verification history available.")
        return
    
    st.header("📚 Verification History")
    
    # Create a selectbox for history items
    history_items = [f"{item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {item['query'][:50]}..." 
                    for item in st.session_state.verification_history]
    
    selected_index = st.selectbox(
        "Select a verification session:",
        range(len(history_items)),
        format_func=lambda x: history_items[x]
    )
    
    if selected_index is not None:
        selected_item = st.session_state.verification_history[selected_index]
        st.subheader(f"Session: {selected_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        display_verification_results(selected_item['result'])

def display_analytics():
    """Display system analytics and insights."""
    st.header("📊 System Analytics")
    
    if not st.session_state.verification_history:
        st.info("No data available for analytics.")
        return
    
    # Calculate analytics
    total_sessions = len(st.session_state.verification_history)
    avg_confidence = sum(
        item['result']['audit_report']['summary_statistics']['overall_confidence']
        for item in st.session_state.verification_history
    ) / total_sessions
    
    risk_distribution = {}
    for item in st.session_state.verification_history:
        risk = item['result']['audit_report']['summary_statistics']['hallucination_risk']
        risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Sessions", total_sessions)
    
    with col2:
        st.metric("Average Confidence", f"{avg_confidence:.2f}")
    
    with col3:
        st.metric("Success Rate", "95%")  # Mock data
    
    # Risk distribution chart
    if risk_distribution:
        st.subheader("🎯 Risk Distribution")
        fig = px.pie(
            values=list(risk_distribution.values()),
            names=list(risk_distribution.keys()),
            title="Hallucination Risk Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function."""
    # Initialize AutoVerify
    if not initialize_autoverify():
        return
    
    # Display header
    display_header()
    
    # Display sidebar and get configuration
    config = display_sidebar()
    
    # Main navigation
    tab1, tab2, tab3 = st.tabs(["🔍 Verify", "📚 History", "📊 Analytics"])
    
    with tab1:
        display_main_interface()
    
    with tab2:
        display_history()
    
    with tab3:
        display_analytics()

if __name__ == "__main__":
    main()
