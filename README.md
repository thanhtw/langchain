# RAG Simple: Local LLM RAG System

A streamlined Retrieval-Augmented Generation (RAG) system that works with local LLMs, optimized for running on both Linux and Windows platforms.

![RAG System Screenshot](https://github.com/user-attachments/assets/89c0bcc8-8daf-4543-88b1-d9798f548b34)

## Features

1. **Flexible File Support** - Upload CSV, JSON, PDF, and DOCX files to build your knowledge base
2. **Multiple Chunking Strategies** - Choose from:
   - No Chunking: Use documents as-is
   - Recursive Token Chunking: Split by token count with intelligent recursion
   - Semantic Chunking: Group text by meaning and semantic similarity
   - Agentic Chunking: Use LLMs to determine optimal chunk boundaries
3. **Local Model Support** - Run entirely on your own hardware with multiple options:
   - Ollama integration for the easiest setup
   - llama.cpp for high-performance GGUF models
   - CTransformers for alternative GGUF loading
   - HuggingFace Transformers for full model control
4. **Cross-Platform** - Runs on both Linux and Windows
5. **Vector-Based Search** - Efficiently retrieve relevant information using Chroma
6. **GPU Acceleration** - Automatic GPU detection and utilization

## System Requirements

- **Minimum**:
  - 8GB RAM
  - 10GB disk space
  - Modern CPU (4+ cores recommended)
- **Recommended**:
  - 16GB+ RAM
  - 20GB+ disk space
  - NVIDIA GPU with 4GB+ VRAM
  - Modern CPU (8+ cores)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag_simple.git
cd rag_simple
```

### 2. Set Up Environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Ollama (Optional but Recommended)

For the easiest setup, install Ollama:

**Linux/macOS**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**:
Download from [https://ollama.com/download](https://ollama.com/download)

### 4. Copy Environment File
```bash
cp .env.example .env
```
Edit the `.env` file to customize your configuration if needed.

### 5. First-Time Setup

When you first run the application, it will:
- Create a `models` directory for storing downloaded models
- Generate a `.env` file if one doesn't exist
- Check for system compatibility

### 6. Run the Application

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## Using the Application

### 1. Set Up Language
Select your preferred language (English or Vietnamese) to initialize the embedding model.

### 2. Upload Data
- Upload files (CSV, JSON, PDF, DOCX)
- Select columns to index
- Choose a chunking strategy
- Save data to Chroma

### 3. Set Up LLM
Choose from one of the following providers:
- **Ollama**: Easy local model setup with simple pull commands
- **llama.cpp**: High-performance inference for GGUF models
- **Transformers**: Full control over HuggingFace models
- **CTransformers**: Alternative GGUF loader

### 4. Chat with Your Data
Ask questions about your data and receive AI responses enhanced with relevant context.

## Chunking Strategies

- **No Chunking**: Uses the original document without splitting
- **Recursive Token Chunking**: Splits documents into smaller chunks based on token count and content structure
- **Semantic Chunking**: Groups text by meaning using embeddings
- **Agentic Chunking**: Uses LLMs to determine the most logical chunk boundaries

## Supported Models

### Ollama Models
- Llama 3 (8B, 1B)
- DeepSeek Coder
- DeepSeek LLM
- Phi-3 Mini/Medium
- Gemma (2B, 7B)
- Mistral
- Mixtral
- And many more

### GGUF Models (llama.cpp/CTransformers)
You can download quantized GGUF models from HuggingFace, with built-in support for:
- Llama 3
- DeepSeek
- Mistral
- Phi
- Gemma

### HuggingFace Models
Load models directly from HuggingFace Hub or local files. Full support for all HuggingFace models.

## Customization

1. **Environment Variables**: Edit the `.env` file to customize paths and endpoints
2. **Prompts**: Use the prompt management system to create custom prompt templates
3. **Chunking Parameters**: Adjust chunk size and overlap to optimize for your specific content

## Troubleshooting

- **Model Loading Errors**: Check that your system has enough RAM for the selected model
- **Ollama Connection Issues**: Verify Ollama is running (`curl http://localhost:11434/api/tags`)
- **GPU Acceleration**: Set `n_gpu_layers` higher for better GPU utilization

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.