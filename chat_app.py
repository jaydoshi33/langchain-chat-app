import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = "langchain-chat-app"

# Page configuration
st.set_page_config(
    page_title="LangChain Chat App",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

def initialize_model():
    """Initialize the OpenAI chat model"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Please set your OPENAI_API_KEY in the .env file")
            return None
        
        model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True,
            api_key=api_key
        )
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def check_langsmith_status():
    """Check LangSmith configuration status"""
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "default")
    
    return {
        "api_key_set": bool(langsmith_api_key),
        "tracing_enabled": langsmith_tracing,
        "project": langsmith_project
    }

def create_system_prompt():
    """Create a system prompt for the chat"""
    return """You are a helpful AI assistant. Keep your responses concise, informative, and helpful."""

def main():
    st.title("🤖 LangChain Chat Application")
    st.markdown("Built with Streamlit and LangChain")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            index=0
        )
        
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness. Lower values make responses more focused."
        )
        
        # System message
        system_message = st.text_area(
            "System Message",
            value=create_system_prompt(),
            height=100,
            help="This message sets the behavior of the AI assistant"
        )
        
        # Clear chat button
        if st.button("Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # LangSmith status
        st.header("🔍 LangSmith Tracing")
        langsmith_status = check_langsmith_status()
        
        if langsmith_status["tracing_enabled"]:
            if langsmith_status["api_key_set"]:
                st.success("✅ LangSmith tracing enabled")
                st.caption(f"Project: {langsmith_status['project']}")
            else:
                st.warning("⚠️ LangSmith API key not set")
                st.caption("Set LANGSMITH_API_KEY in .env file")
        else:
            st.info("ℹ️ LangSmith tracing disabled")
        
        if st.button("View LangSmith Dashboard", type="secondary"):
            st.markdown("[Open LangSmith Dashboard](https://smith.langchain.com/)")
    
    # Initialize model if not already done
    if st.session_state.model is None:
        st.session_state.model = initialize_model()
    
    if st.session_state.model is None:
        st.stop()
    
    # Update model parameters
    st.session_state.model.temperature = temperature
    st.session_state.model.model_name = model_name
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to chat about?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Create messages for the model
                messages = [SystemMessage(content=system_message)]
                
                # Add chat history
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                
                # Stream the response
                for chunk in st.session_state.model.stream(messages):
                    if chunk.content:
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                full_response = "Sorry, I encountered an error. Please try again."
                message_placeholder.markdown(full_response)
    
    # Display token usage if available
    if st.session_state.messages:
        with st.expander("Session Info"):
            st.write(f"Total messages: {len(st.session_state.messages)}")
            st.write(f"Model: {model_name}")
            st.write(f"Temperature: {temperature}")

if __name__ == "__main__":
    main()
