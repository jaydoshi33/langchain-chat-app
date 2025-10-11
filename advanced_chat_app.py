import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json

# Load environment variables
load_dotenv()

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = "langchain-advanced-chat"

# Page configuration
st.set_page_config(
    page_title="Advanced LangChain Chat",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

if "chain" not in st.session_state:
    st.session_state.chain = None

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

def create_prompt_template():
    """Create a chat prompt template"""
    template = """You are a helpful AI assistant with the following characteristics:
    
    - You are knowledgeable, friendly, and professional
    - You provide accurate and helpful information
    - You ask clarifying questions when needed
    - You maintain context throughout the conversation
    - You can help with various topics including coding, general knowledge, and problem-solving
    
    Current conversation context:
    {context}
    
    User: {input}
    Assistant:"""
    
    return ChatPromptTemplate.from_template(template)

def create_chain(model, prompt_template):
    """Create a LangChain chain"""
    chain = (
        {"context": RunnablePassthrough(), "input": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )
    return chain

def get_conversation_context():
    """Get the conversation context from recent messages"""
    if len(st.session_state.messages) <= 2:
        return "This is the beginning of our conversation."
    
    # Get last few messages for context
    recent_messages = st.session_state.messages[-4:]
    context = "Recent conversation:\n"
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"
    
    return context

def main():
    st.title("🚀 Advanced LangChain Chat Application")
    st.markdown("Built with Streamlit, LangChain, and OpenAI")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
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
        
        # Max tokens
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Maximum number of tokens in the response"
        )
        
        # Conversation mode
        conversation_mode = st.selectbox(
            "Conversation Mode",
            ["Normal", "Creative", "Technical", "Casual"],
            index=0
        )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Export chat button
        if st.button("📥 Export Chat"):
            export_chat()
    
    # Initialize model and chain
    if st.session_state.model is None:
        st.session_state.model = initialize_model()
    
    if st.session_state.model is None:
        st.stop()
    
    # Update model parameters
    st.session_state.model.temperature = temperature
    st.session_state.model.model_name = model_name
    st.session_state.model.max_tokens = max_tokens
    
    # Create or update chain
    prompt_template = create_prompt_template()
    st.session_state.chain = create_chain(st.session_state.model, prompt_template)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
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
                    # Get conversation context
                    context = get_conversation_context()
                    
                    # Stream the response using the chain
                    for chunk in st.session_state.chain.stream({
                        "context": context,
                        "input": prompt
                    }):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    full_response = "Sorry, I encountered an error. Please try again."
                    message_placeholder.markdown(full_response)
    
    with col2:
        st.header("📊 Chat Stats")
        st.metric("Total Messages", len(st.session_state.messages))
        st.metric("User Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
        st.metric("AI Messages", len([m for m in st.session_state.messages if m["role"] == "assistant"]))
        
        st.header("🔧 Model Info")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Temperature:** {temperature}")
        st.write(f"**Max Tokens:** {max_tokens}")
        st.write(f"**Mode:** {conversation_mode}")

def export_chat():
    """Export chat history to JSON"""
    if st.session_state.messages:
        chat_data = {
            "messages": st.session_state.messages,
            "export_timestamp": str(st.session_state.get("export_time", "unknown"))
        }
        
        st.download_button(
            label="Download Chat History",
            data=json.dumps(chat_data, indent=2),
            file_name="chat_history.json",
            mime="application/json"
        )
    else:
        st.warning("No messages to export")

if __name__ == "__main__":
    main()
