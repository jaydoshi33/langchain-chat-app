# LangChain Chat Application

A modern chat application built with Streamlit and LangChain, featuring OpenAI integration and real-time streaming responses.

## Features

- 🤖 **AI Chat Interface**: Interactive chat with OpenAI models
- ⚡ **Real-time Streaming**: See responses as they're generated
- 🎛️ **Configurable Settings**: Adjust temperature, model, and other parameters
- 📱 **Responsive Design**: Clean, modern UI with Streamlit
- 🔗 **LangChain Integration**: Built on LangChain for extensibility
- 📊 **Chat Statistics**: Track conversation metrics
- 💾 **Export Functionality**: Save chat history
- 🔍 **LangSmith Tracing**: Built-in observability and debugging (optional)

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

### 3. Run the Application

**Basic Version:**
```bash
streamlit run chat_app.py
```

**Advanced Version:**
```bash
streamlit run advanced_chat_app.py
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

- `streamlit>=1.28.0` - Web application framework
- `langchain>=0.3.0` - LangChain core library
- `langchain-openai>=0.2.0` - OpenAI integration for LangChain
- `python-dotenv>=1.0.0` - Environment variable management
- `openai>=1.0.0` - OpenAI Python client

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

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your OpenAI API key is correctly set in the `.env` file
2. **Import Errors**: Run `pip install -r requirements.txt` to install dependencies
3. **Streamlit Issues**: Ensure Streamlit is properly installed and updated

### Getting Help

- Check the [LangChain Documentation](https://python.langchain.com/docs/)
- Review the [Streamlit Documentation](https://docs.streamlit.io/)
- OpenAI API Documentation: https://platform.openai.com/docs

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
