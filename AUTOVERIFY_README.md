# 🔍 AutoVerify - AI Agent for Hallucination Detection and Fact Verification

## 🎯 Project Overview

AutoVerify is a sophisticated multi-agent AI system designed to detect, verify, and correct hallucinations in generative AI systems. It transforms unreliable AI outputs into trustworthy, fact-checked responses through autonomous verification and correction.

## 🚨 Problem Statement

Modern Generative AI chatbots frequently produce hallucinations — confidently incorrect or unverifiable statements — especially when:
- Asked domain-specific questions (finance, healthcare, legal, etc.)
- Relying on outdated or incomplete training data
- Summarizing long, multi-source documents

This leads to loss of user trust, misinformed decisions, and regulatory risks in enterprise settings.

## ✅ Solution: AutoVerify Agent System

AutoVerify uses a multi-agent architecture to cross-check, retrieve, and validate information dynamically, providing explainable verification reports for enterprise use.

## 🏗️ System Architecture

### 1. **Generator Agent (LLM Core)**
- Takes user queries and produces initial responses
- Supports multiple models (GPT-4, Claude, Llama, etc.)
- Configurable temperature and response styles
- Passes responses to verification layer

### 2. **Verifier Agent**
- Parses generated answers into claim units
- Uses RAG to search authoritative sources (Wikipedia, PubMed, etc.)
- Assigns credibility scores (0–1) based on:
  - Source reliability
  - Content match ratio
  - Factual consistency
- Detects potential hallucinations

### 3. **Correction Agent**
- Re-generates or rephrases content with low confidence scores
- Uses verified data to improve accuracy
- Provides inline citations and supporting evidence
- Maintains original intent while improving factual accuracy

### 4. **Audit & Feedback Agent**
- Logs all verification steps to structured database
- Provides explainable "Verification Reports" for enterprise use
- Tracks system performance and accuracy metrics
- Generates insights and recommendations

## 🚀 Features

### Core Capabilities
- **Multi-Agent Architecture**: Coordinated verification pipeline
- **Real-time Fact Checking**: Instant verification of AI responses
- **Source Verification**: Cross-referencing with authoritative sources
- **Automatic Correction**: Self-improving content generation
- **Comprehensive Auditing**: Complete audit trails and reporting
- **Enterprise Ready**: Compliance and regulatory support

### Advanced Features
- **RAG Integration**: Retrieval-Augmented Generation for source verification
- **Confidence Scoring**: Quantitative assessment of response reliability
- **Hallucination Detection**: Automated identification of false information
- **Citation Management**: Automatic source attribution
- **Performance Analytics**: System monitoring and optimization
- **Multi-Model Support**: Works with various LLM providers

## 📁 Project Structure

```
autoverify/
├── __init__.py                 # Package initialization
├── core/
│   ├── base_agent.py          # Base agent class
│   └── orchestrator.py        # Multi-agent coordination
├── agents/
│   ├── generator_agent.py     # LLM response generation
│   ├── verifier_agent.py      # Fact verification and RAG
│   ├── correction_agent.py    # Content correction
│   └── audit_agent.py         # Logging and reporting
└── autoverify_app.py          # Streamlit web interface
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Optional: Anthropic API key for Claude models

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/jaydoshi33/langchain-chat-app.git
cd langchain-chat-app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
# Copy the template
cp env_template.txt .env

# Edit .env with your API keys
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=autoverify
LANGSMITH_TRACING=true
```

4. **Run the application**
```bash
streamlit run autoverify_app.py
```

## 🎮 Usage

### Web Interface

1. **Access the application** at `http://localhost:8501`
2. **Configure settings** in the sidebar:
   - Verification level (basic, standard, thorough)
   - Model selection and parameters
   - Confidence thresholds
3. **Enter your query** in the main interface
4. **Review results**:
   - Generated response
   - Verification details
   - Corrections applied
   - Audit report

### Programmatic Usage

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

## 📊 Verification Process

### 1. Generation Phase
- User query processed by Generator Agent
- Initial response generated using selected LLM
- Response metadata and confidence indicators captured

### 2. Verification Phase
- Verifier Agent extracts factual claims
- RAG system searches authoritative sources
- Credibility scores assigned to each claim
- Hallucination risk assessment performed

### 3. Correction Phase
- Correction Agent identifies low-confidence content
- Content re-generated using verified information
- Citations and evidence added
- Confidence scores improved

### 4. Audit Phase
- Audit Agent logs all activities
- Comprehensive report generated
- Performance metrics calculated
- Compliance status assessed

## 🔧 Configuration

### Agent Configuration

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

### Verification Levels

- **Basic**: Quick verification with minimal sources
- **Standard**: Balanced verification with moderate source coverage
- **Thorough**: Comprehensive verification with extensive source checking

## 📈 Performance Metrics

### System Metrics
- **Processing Time**: End-to-end verification duration
- **Confidence Scores**: Average confidence across all claims
- **Verification Rate**: Percentage of claims successfully verified
- **Correction Rate**: Percentage of responses requiring correction

### Quality Metrics
- **Hallucination Detection**: Accuracy of false information identification
- **Source Reliability**: Quality assessment of verification sources
- **Citation Accuracy**: Correctness of added citations
- **User Satisfaction**: Feedback-based quality assessment

## 🏢 Enterprise Features

### Compliance & Auditing
- **Complete Audit Trails**: Every verification step logged
- **Regulatory Compliance**: Meets enterprise compliance requirements
- **Source Traceability**: Full traceability of all information sources
- **Performance Reporting**: Detailed analytics and insights

### Security & Privacy
- **API Key Management**: Secure handling of authentication credentials
- **Data Retention**: Configurable data retention policies
- **Access Control**: Role-based access to verification reports
- **Privacy Protection**: No storage of sensitive user data

## 🔍 API Reference

### Orchestrator Class

```python
class AutoVerifyOrchestrator:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    async def process_query(self, query: str, context: Dict[str, Any] = None, verification_level: str = "standard") -> Dict[str, Any]
    def get_system_status(self) -> Dict[str, Any]
    def get_agent_capabilities(self) -> Dict[str, List[str]]
    def reset_system(self) -> None
    def configure_agent(self, agent_id: str, config: Dict[str, Any]) -> bool
```

### Agent Classes

All agents inherit from `BaseAgent` and implement:
- `process_message(message: AgentMessage) -> AgentMessage`
- `get_capabilities() -> List[str]`
- `get_status() -> Dict[str, Any]`

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/unit/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### End-to-End Tests
```bash
python -m pytest tests/e2e/
```

## 📚 Examples

### Basic Verification
```python
# Simple fact verification
result = await orchestrator.process_query(
    "What is the population of Tokyo?",
    verification_level="standard"
)
```

### Domain-Specific Verification
```python
# Medical information verification
result = await orchestrator.process_query(
    "What are the side effects of aspirin?",
    context={"domain": "medical", "audience": "healthcare_professionals"},
    verification_level="thorough"
)
```

### Financial Information Verification
```python
# Financial data verification
result = await orchestrator.process_query(
    "What is the current interest rate set by the Federal Reserve?",
    context={"domain": "finance", "currency": "USD"},
    verification_level="thorough"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangChain team for the excellent framework
- OpenAI for providing powerful language models
- Streamlit for the intuitive web interface
- The open-source community for various dependencies

## 📞 Support

For support, email support@autoverify.ai or join our Slack channel.

## 🔮 Roadmap

### Phase 1 (Current)
- ✅ Multi-agent architecture
- ✅ Basic verification pipeline
- ✅ Web interface
- ✅ Audit logging

### Phase 2 (Next)
- 🔄 Advanced RAG with vector databases
- 🔄 Real-time source updates
- 🔄 Custom knowledge base integration
- 🔄 API endpoints

### Phase 3 (Future)
- 📋 Multi-language support
- 📋 Voice interface
- 📋 Mobile application
- 📋 Enterprise integrations

---

**AutoVerify** - Making AI responses trustworthy, one verification at a time. 🔍✨
