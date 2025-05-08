# RagChitChat: Retrieval Augmented Chat for CTSE Lecture Notes

![RagChitChat Banner](https://img.shields.io/badge/RagChitChat-A%20Local%20RAG%20Chatbot-blue)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-green)](https://ollama.ai)
[![Haystack](https://img.shields.io/badge/RAG-Haystack-orange)](https://haystack.deepset.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Retrieval Augmented Generation (RAG) chatbot designed specifically for CTSE (Current Trends in Software Engineering) lecture notes. It combines local LLM capabilities with advanced document retrieval to provide accurate, context-aware answers about software engineering concepts.

## 🌟 Features

- **📚 Document Processing**: Intelligent handling of various file formats
  - PDF lecture notes with multi-page support
  - PowerPoint (PPTX) presentations
  - Automatic chunking with overlap for context preservation
  - Text extraction with metadata retention (source, page numbers)

- **🔍 Advanced Retrieval**:
  - Hybrid search combining semantic (vector) and keyword-based (BM25) retrieval
  - ChromaDB vector database for efficient similarity search
  - Custom relevance scoring and result ranking
  - Configurable retrieval parameters (chunk size, overlap, top_k)

- **🧠 Local LLM Integration**:
  - Runs entirely offline with Ollama
  - Multiple model support with live switching (mistral, llama2, etc.)
  - Optimized prompting with context window management
  - Stream responses for better user experience

- **📊 Advanced Prompt Engineering**:
  - Chain-of-thought reasoning
  - Few-shot learning examples for complex queries
  - Self-reflective answer generation
  - Structured output formatting based on query type
  - Dynamic prompt selection based on question analysis

- **💻 Rich Terminal UI**:
  - Beautiful interactive interface with Rich
  - Markdown rendering for formatted responses
  - Progress indicators during processing
  - Command system with help, history, and model management
  - Syntax highlighting and pretty printing

## 📋 Project Structure(OOP Based)

```
RagChitChat/
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup for imports
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
│
├── data/                    # Raw lecture notes (.pdf, .pptx)
├── processed/               # Processed text data
├── chroma_db/               # Vector database storage
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── document_processor/  # PDF and PPTX processing
│   │   ├── __init__.py
│   │   └── processor.py     # Document processing classes
│   │
│   ├── vector_store/        # ChromaDB integration
│   │   ├── __init__.py
│   │   └── chroma_store.py  # Vector database management
│   │
│   ├── retriever/           # Haystack retrieval components
│   │   ├── __init__.py
│   │   └── haystack_retriever.py  # Document retrieval
│   │
│   ├── llm/                 # Ollama integration
│   │   ├── __init__.py
│   │   └── ollama_client.py # LLM interface
│   │
│   ├── prompts/             # Prompt engineering
│   │   ├── __init__.py
│   │   └── prompt_templates.py  # Advanced prompt templates
│   │
│   ├── interface/           # Rich terminal UI
│   │   ├── __init__.py
│   │   └── terminal_ui.py   # Terminal interface
│   │
│   └── main.py              # Entry point
│
└── config/                  # Configuration files
    ├── __init__.py
    └── settings.py          # Application settings
```

## 🛠️ Requirements

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for larger models)
- Storage space for models and vector database (2GB+)
- Windows, macOS, or Linux

### Required Tools
- **Ollama**: For running local LLMs ([Download](https://ollama.ai/))
- **Python**: 3.8+ ([Download](https://www.python.org/downloads/))

## 🚀 Complete Setup Instructions

### 1. Install Python and Setup Virtual Environment

```bash
# Install Python (if not already installed)
# Download from https://www.python.org/downloads/

# Clone the repository (if using git)
git clone https://github.com/nxdun/RagChitChat.git
cd RagChitChat

# Create and activate virtual environment
python -m venv env

# On Windows
.\env\Scripts\activate

# On macOS/Linux
source env/bin/activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# (optional)Install package for imports (optional, if import issues occur)
pip install -e .
```

### 3. Install and Configure Ollama

```bash
# Download and install Ollama from https://ollama.ai/

# Pull required models (after installing)
ollama pull mistral:7b-instruct-v0.3-q4_1  # Default model(You can edit this on env)
ollama pull llama2                          # Alternative model (optional)
```

### 4. Prepare Your Data

```bash
# Create data directory (root Dir)
mkdir -p data

# Copy your lecture notes into data folder
# Supported formats: PDF, PPTX
# Example: cp ~/Downloads/CTSE_Lecture*.pdf data/
```

### 5. Configure Environment Variables (Optional : Or Use Defaults)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings
# nano .env or use any text editor
```

### 6. Run the Application

```bash
# Start the chatbot
python src/main.py
```

## 📝 Configuration Options

You can configure the application through environment variables or by editing `config/settings.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `RAGCHITCHAT_DATA_DIR` | Directory containing lecture notes | `data` |
| `RAGCHITCHAT_PROCESSED_DIR` | Directory for processed documents | `processed` |
| `RAGCHITCHAT_DB_DIR` | Directory for vector database | `chroma_db` |
| `RAGCHITCHAT_MODEL` | Default Ollama model | `mistral:7b-instruct-v0.3-q4_1` |
| `OLLAMA_URL` | Ollama API URL | `http://localhost:11434` |
| `RAGCHITCHAT_CHUNK_SIZE` | Document chunk size | `1000` |
| `RAGCHITCHAT_CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `RAGCHITCHAT_TOP_K` | Number of context documents to retrieve | `5` |

## 🖥️ Usage Guide

### Basic Commands

Once the application is running, you can interact with it through the terminal:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/exit` | Exit the application |
| `/clear` | Clear conversation history |
| `/history` | Show conversation history |
| `/models` | List available models |
| `/model <name>` | Switch to a different model |
| `/info` | Show system information |
| `/about` | About the application |

### Example Questions

Here are some examples of questions you can ask:

- "What is continuous integration?"
- "Explain the difference between DevOps and DevSecOps"
- "What are the benefits of microservices architecture?"
- "How does containerization improve software deployment?"
- "Compare agile and waterfall methodologies in software engineering"

## 📊 Advanced Features

### Model Switching

You can switch between different LLMs during runtime:

```
/models                         # List available models
/model mistral:7b-instruct-v0.3-q4_1   # Switch to Mistral
/model llama2                   # Switch to Llama 2
```

### Dynamic Prompt Selection

The system automatically selects prompting strategies based on question type:
- Factual questions: Standard RAG with direct answers
- Comparative questions: Structured comparison format
- Procedural questions: Step-by-step instructions
- Complex questions: Self-reflective generation

### Hybrid Search

The retrieval system combines two search methods for better results:
1. **Vector search**: Semantic similarity using embeddings
2. **BM25 search**: Keyword-based traditional search

## 🔧 Troubleshooting

### Common Issues and Solutions

**Problem**: Import errors when running `python src/main.py`
**Solution**: Install the package in development mode:
```bash
pip install -e .
```

**Problem**: "Cannot connect to Ollama" error
**Solution**: Ensure Ollama is installed and running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
# If not running, start Ollama application
```

**Problem**: Model not found when switching models
**Solution**: Pull the model using Ollama CLI:
```bash
ollama pull <model_name>
```

**Problem**: High memory usage
**Solution**: Use a smaller model or reduce `RAGCHITCHAT_TOP_K` in settings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Ollama](https://ollama.ai/) for local LLM capabilities
- [Haystack](https://haystack.deepset.ai/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Rich](https://rich.readthedocs.io/) for the terminal UI
- All open-source contributors to the libraries used
