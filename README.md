# LangChain Chat Application & AutoVerify System

A comprehensive suite of AI applications built with Streamlit and LangChain, featuring OpenAI integration, real-time streaming responses, and advanced hallucination detection through the AutoVerify multi-agent system.

## Features

### Chat Applications
- 🤖 **AI Chat Interface**: Interactive chat with OpenAI models
- ⚡ **Real-time Streaming**: See responses as they're generated
- 🎛️ **Configurable Settings**: Adjust temperature, model, and other parameters
- 📱 **Responsive Design**: Clean, modern UI with Streamlit
- 🔗 **LangChain Integration**: Built on LangChain for extensibility
- 📊 **Chat Statistics**: Track conversation metrics
- 💾 **Export Functionality**: Save chat history
- 🔍 **LangSmith Tracing**: Built-in observability and debugging (optional)

### AutoVerify System
- 🔍 **Hallucination Detection**: Advanced AI agent system for fact verification
- 🤖 **Multi-Agent Architecture**: Coordinated verification pipeline with specialized agents
- 📊 **Real-time Fact Checking**: Instant verification of AI responses against authoritative sources
- ✏️ **Automatic Correction**: Self-improving content generation with verified information
- 📋 **Comprehensive Auditing**: Complete audit trails and compliance reporting
- 🎯 **Confidence Scoring**: Quantitative assessment of response reliability
- 📚 **RAG Integration**: Retrieval-Augmented Generation for source verification
- 🏢 **Enterprise Ready**: Compliance and regulatory support features

## Quick Start

### 1. Setup

Run the setup script to install dependencies and create configuration files:

```bash
python setup.py
```

### 2. Configure API Keys

Edit the `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=default
LANGSMITH_TRACING=true
```

**Required:**
- Get your OpenAI API key from: https://platform.openai.com/api-keys

**Optional (for tracing):**
- Get your LangSmith API key from: https://smith.langchain.com/
- LangSmith provides observability and debugging for your LangChain applications

### 3. Run the Applications

**Basic Chat App:**
```bash
streamlit run chat_app.py
```

**Advanced Chat App:**
```bash
streamlit run advanced_chat_app.py
```

**AutoVerify System:**
```bash
streamlit run autoverify_app.py
```

## Applications

### Basic Chat App (`chat_app.py`)
- Simple chat interface
- Model selection (GPT-3.5, GPT-4)
- Temperature control
- Custom system messages
- Chat history management

### Advanced Chat App (`advanced_chat_app.py`)
- Enhanced UI with statistics
- LangChain prompt templates
- Conversation context awareness
- Export functionality
- Multiple conversation modes
- Token usage tracking

### AutoVerify System (`autoverify_app.py`)
- **Multi-Agent Architecture**: Coordinated verification pipeline with specialized agents
- **Generator Agent**: Produces initial AI responses using configurable models
- **Verifier Agent**: Extracts claims and verifies them against authoritative sources using RAG
- **Correction Agent**: Automatically corrects low-confidence content with verified information
- **Audit Agent**: Provides comprehensive logging and compliance reporting
- **Real-time Verification**: Instant fact-checking with confidence scoring
- **Source Attribution**: Automatic citation and evidence management
- **Enterprise Features**: Audit trails, compliance reporting, and performance analytics

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `LANGSMITH_API_KEY` | LangSmith API key for tracing | No |
| `LANGSMITH_PROJECT` | LangSmith project name | No |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | No |

### Model Settings

- **Model Selection**: Choose between GPT-3.5-turbo, GPT-4, or GPT-4-turbo-preview
- **Temperature**: Control response randomness (0.0 - 2.0)
- **Max Tokens**: Limit response length
- **System Message**: Customize AI behavior

## LangChain Integration

This application demonstrates key LangChain concepts:

- **Chat Models**: Using `ChatOpenAI` for model interaction
- **Prompt Templates**: Structured prompt management with `ChatPromptTemplate`
- **Chains**: Creating reusable processing pipelines
- **Streaming**: Real-time response generation
- **Message Types**: Proper handling of `HumanMessage`, `AIMessage`, and `SystemMessage`

## AutoVerify System

### Overview

AutoVerify is a sophisticated multi-agent AI system designed to detect, verify, and correct hallucinations in generative AI systems. It transforms unreliable AI outputs into trustworthy, fact-checked responses through autonomous verification and correction.

### Problem Statement

Modern Generative AI chatbots frequently produce hallucinations — confidently incorrect or unverifiable statements — especially when:
- Asked domain-specific questions (finance, healthcare, legal, etc.)
- Relying on outdated or incomplete training data
- Summarizing long, multi-source documents

This leads to loss of user trust, misinformed decisions, and regulatory risks in enterprise settings.

### Solution: Multi-Agent Architecture

AutoVerify uses a coordinated multi-agent system to cross-check, retrieve, and validate information dynamically:

#### 1. Generator Agent
- Takes user queries and produces initial responses
- Supports multiple models (GPT-4, Claude, Llama, etc.)
- Configurable temperature and response styles
- Passes responses to verification layer

#### 2. Verifier Agent
- Parses generated answers into claim units
- Uses RAG to search authoritative sources (Wikipedia, PubMed, etc.)
- Assigns credibility scores (0–1) based on:
  - Source reliability
  - Content match ratio
  - Factual consistency
- Detects potential hallucinations

#### 3. Correction Agent
- Re-generates or rephrases content with low confidence scores
- Uses verified data to improve accuracy
- Provides inline citations and supporting evidence
- Maintains original intent while improving factual accuracy

#### 4. Audit Agent
- Logs all verification steps to structured database
- Provides explainable "Verification Reports" for enterprise use
- Tracks system performance and accuracy metrics
- Generates insights and recommendations

### Verification Process

1. **Generation Phase**: User query processed by Generator Agent
2. **Verification Phase**: Verifier Agent extracts claims and searches authoritative sources
3. **Correction Phase**: Correction Agent improves low-confidence content
4. **Audit Phase**: Audit Agent logs activities and generates comprehensive reports

### Usage Examples

#### Basic Verification
```python
from autoverify.core.orchestrator import AutoVerifyOrchestrator

# Initialize the system
orchestrator = AutoVerifyOrchestrator()

# Process a query
result = await orchestrator.process_query(
    query="What is the capital of France?",
    verification_level="standard"
)

# Access results
print(f"Generated: {result['generated_response']['text']}")
print(f"Confidence: {result['audit_report']['summary_statistics']['overall_confidence']}")
```

#### Domain-Specific Verification
```python
# Medical information verification
result = await orchestrator.process_query(
    query="What are the side effects of aspirin?",
    context={"domain": "medical", "audience": "healthcare_professionals"},
    verification_level="thorough"
)
```

### Configuration Options

#### Verification Levels
- **Basic**: Quick verification with minimal sources
- **Standard**: Balanced verification with moderate source coverage
- **Thorough**: Comprehensive verification with extensive source checking

#### Agent Configuration
```python
config = {
    "generator": {
        "primary_model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "verifier": {
        "confidence_threshold": 0.7,
        "max_sources": 5,
        "embedding_model": "text-embedding-3-small"
    },
    "corrector": {
        "correction_model": "gpt-4",
        "confidence_threshold": 0.7,
        "citation_style": "inline"
    },
    "auditor": {
        "database_path": "autoverify_audit.db",
        "retention_days": 90
    }
}
```

### Enterprise Features

- **Complete Audit Trails**: Every verification step logged
- **Regulatory Compliance**: Meets enterprise compliance requirements
- **Source Traceability**: Full traceability of all information sources
- **Performance Reporting**: Detailed analytics and insights
- **API Integration**: Programmatic access to verification services

## LangSmith Tracing

LangSmith provides powerful observability and debugging capabilities for your LangChain applications:

### What LangSmith Tracks
- **Token Usage**: Monitor input/output tokens and costs
- **Latency**: Track response times for each request
- **Model Parameters**: Temperature, model version, etc.
- **Prompt Engineering**: See exactly what prompts are sent to the model
- **Error Tracking**: Debug failed requests and errors
- **Chain Execution**: Visualize the flow of your LangChain chains

### Setting Up LangSmith
1. Sign up at [smith.langchain.com](https://smith.langchain.com/)
2. Get your API key from the settings page
3. Add it to your `.env` file:
   ```env
   LANGSMITH_API_KEY=your_api_key_here
   LANGSMITH_PROJECT=your_project_name
   LANGSMITH_TRACING=true
   ```

### Viewing Traces
- The chat app shows LangSmith status in the sidebar
- Click "View LangSmith Dashboard" to open the web interface
- Each conversation will appear as a trace in your project
- You can see detailed information about each LLM call

## Dependencies

### Core Dependencies
- `streamlit>=1.28.0` - Web application framework
- `langchain>=0.3.0` - LangChain core library
- `langchain-openai>=0.2.0` - OpenAI integration for LangChain
- `langchain-anthropic>=0.3.0` - Anthropic Claude integration
- `langchain-community>=0.3.0` - Community integrations and tools
- `python-dotenv>=1.0.0` - Environment variable management
- `openai>=1.0.0` - OpenAI Python client

### AutoVerify-Specific Dependencies
- `plotly>=5.0.0` - Interactive visualizations and analytics
- `faiss-cpu>=1.7.0` - Vector similarity search for RAG functionality

## Usage Examples

### Basic Chat
```python
# The app handles message formatting automatically
# Just type your message and press Enter
```

### Custom System Prompts
```python
# Use the sidebar to customize the AI's behavior
# Example: "You are a helpful coding assistant"
```

### Export Chat History
```python
# Click "Export Chat" in the advanced app
# Downloads a JSON file with your conversation
```

### AutoVerify Usage

#### Web Interface
1. Access the AutoVerify app at `http://localhost:8501`
2. Configure verification settings in the sidebar
3. Enter your query for verification
4. Review the generated response, verification details, and corrections

#### Programmatic Usage
```python
from autoverify.core.orchestrator import AutoVerifyOrchestrator

# Initialize with custom configuration
config = {
    "generator": {"primary_model": "gpt-4", "temperature": 0.7},
    "verifier": {"confidence_threshold": 0.8, "max_sources": 3},
    "corrector": {"correction_model": "gpt-4"},
    "auditor": {"database_path": "audit.db"}
}

orchestrator = AutoVerifyOrchestrator(config)

# Process a query with verification
result = await orchestrator.process_query(
    query="What is the current inflation rate in the US?",
    verification_level="thorough"
)

# Access verification results
print(f"Confidence Score: {result['audit_report']['summary_statistics']['overall_confidence']}")
print(f"Hallucination Risk: {result['audit_report']['summary_statistics']['hallucination_risk']}")
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your OpenAI API key is correctly set in the `.env` file
2. **Import Errors**: Run `pip install -r requirements.txt` to install dependencies
3. **Streamlit Issues**: Ensure Streamlit is properly installed and updated
4. **AutoVerify Initialization Error**: Check that all required dependencies are installed, especially `faiss-cpu` and `plotly`
5. **Database Connection Issues**: Ensure write permissions for the audit database file
6. **Memory Issues**: For large verification tasks, consider reducing `max_sources` in verifier configuration

### AutoVerify-Specific Issues

1. **Verification Timeout**: Increase timeout settings or reduce verification level
2. **Low Confidence Scores**: Adjust confidence thresholds or increase source coverage
3. **RAG Search Failures**: Check internet connectivity for source retrieval
4. **Agent Communication Errors**: Restart the application to reinitialize agents

### Getting Help

- Check the [LangChain Documentation](https://python.langchain.com/docs/)
- Review the [Streamlit Documentation](https://docs.streamlit.io/)
- OpenAI API Documentation: https://platform.openai.com/docs
- AutoVerify Documentation: See `AUTOVERIFY_README.md` for detailed system documentation

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
