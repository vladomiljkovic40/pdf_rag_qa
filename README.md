# Enhanced PDF RAG QA System

🤖 **Production-Ready RAG Pipeline for PDF Document Question-Answering**

A comprehensive Retrieval-Augmented Generation (RAG) system that processes PDF documents and provides intelligent answers using vector similarity search with multiple AI generation backends.

## 🎯 RAG Architecture Overview

This system implements a complete **RAG (Retrieval-Augmented Generation) pipeline** with enterprise-grade features:

1. **Document Ingestion** - PDF text extraction and preprocessing using PyMuPDF
2. **Text Segmentation** - Intelligent chunking with configurable size and overlap
3. **Vector Embeddings** - Semantic embeddings using SentenceTransformers
4. **Vector Database** - FAISS index with cosine similarity search
5. **Retrieval System** - Top-k relevant document segments with confidence scoring
6. **Generation Engine** - Multiple AI backends (OpenAI API, Local LLMs, Rule-based)
7. **Interactive Interface** - Real-time model switching and query processing

## ⚡ Quick Start

### 1. Installation
```bash
# Install core dependencies
pip install pymupdf sentence-transformers faiss-cpu numpy torch transformers accelerate

# For OpenAI integration (optional)
pip install openai
```

### 2. Environment Setup (Optional)
```bash
# For OpenAI API support
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 3. Run the RAG System
```bash
# Interactive mode
python pdf_rag_qa.py

# Direct PDF loading
python pdf_rag_qa.py "path/to/document.pdf"
```

## 🧠 Local LLM Model Selection

The system provides **intelligent model control** to prevent unwanted downloads. Models are only downloaded if explicitly enabled:

### **Available Models:**
```python
# Edit these settings in pdf_rag_qa.py to control which models are downloaded:
available_models = {
    "distilgpt2": {
        "name": "DistilGPT2",
        "size": "328MB", 
        "enabled": True    # ✅ Will download and use
    },
    "gpt2": {
        "name": "GPT-2",
        "size": "548MB",
        "enabled": False   # ❌ Won't download (change to True if wanted)
    },
    "microsoft/DialoGPT-large": {
        "name": "DialoGPT Large", 
        "size": "1.1GB",
        "enabled": False   # ❌ Large model, enable only if you want quality
    }
}
```

### **Model Management Commands:**
```bash
models                  # Show available loaded models
model 1                 # Switch to model by number
model DistilGPT2       # Switch by name
model distilgpt2       # Switch by ID
```

## 🎮 Interactive RAG Commands

### **Core Commands:**
```bash
# Vector search and generation
<question>              # Ask any question about the document

# Model management  
models                  # Show available local models
model <name/number>     # Switch generation model
mode <type>            # Change generation mode

# System information
info                   # Show document and system status  
help                   # Display all commands
quit                   # Exit the system
```

### **Generation Modes:**
```bash
mode auto       # OpenAI → Local LLM → Rules (default)
mode local      # Local LLM only (privacy mode)  
mode openai     # OpenAI API only (requires key)
mode hybrid     # Try multiple methods
```

## 📊 Model Performance Comparison

| Model | Size | Speed | Quality | Memory | Best For |
|-------|------|-------|---------|---------|----------|
| **DistilGPT2** | 328MB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 1GB | Quick questions |
| **GPT-2** | 548MB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 1.5GB | Balanced use |
| **DialoGPT Small** | 117MB | ⭐⭐⭐⭐⭐ | ⭐⭐ | 0.5GB | Lightweight |
| **DialoGPT Medium** | 355MB | ⭐⭐⭐ | ⭐⭐⭐⭐ | 1GB | Good balance |
| **DialoGPT Large** | 1.1GB | ⭐⭐ | ⭐⭐⭐⭐⭐ | 3GB | Best quality |

## 🔍 RAG Session Example

```
Enhanced PDF RAG QA System - Interactive Mode
============================================

Loading 2 selected models...
SUCCESS: DistilGPT2 loaded successfully!  
Default model: DistilGPT2

Enter PDF file path: research_paper.pdf

Loading PDF: research_paper.pdf
Extracted 25 pages
Created 89 text segments  
Vector index built successfully!

Document Info:
 File: research_paper.pdf
 Pages: 25
 Segments: 89

AI Capabilities:
 OpenAI Available: NO
 Local LLM Loaded: YES
 Available Models: 2
 Current Model: DistilGPT2
 Generation Mode: auto

Question: What are the main findings?

Processing question: What are the main findings?
Using model: DistilGPT2

==================================================
ANSWER (LOCAL_LLM (DISTILGPT2)):
==================================================
Based on the document:

The main findings include significant performance improvements 
in the proposed method, showing 15% better accuracy compared to 
baseline approaches. The system also demonstrated improved 
efficiency with 40% reduction in processing time while 
maintaining quality standards.

Sources (Confidence: 87%):
 1. Page 12 (similarity: 94%)
    Preview: Our experimental results show a 15% improvement...
 2. Page 18 (similarity: 86%)  
    Preview: Processing efficiency increased by 40%...

Generated using: local_llm (DistilGPT2) method
--------------------------------------------------
```

## 🛠️ System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended for large models)
- **CPU**: Intel i3/Ryzen 3 or better  
- **Storage**: 2-5GB for models and embeddings
- **Internet**: Required for initial model downloads only

## 🔧 Configuration Options

### **Text Processing:**
```python
# Initialize with custom parameters
qa = PDFRagQA(
    embedding_model="all-MiniLM-L6-v2",  # Embedding model
    chunk_size=400,                       # Words per segment
    overlap_size=50,                      # Overlap between chunks
    generation_mode="auto"                # Generation strategy
)
```

### **Model Control:**
To prevent unwanted model downloads, edit the `enabled` field in `pdf_rag_qa.py`:

```python
# Only enabled models will be downloaded
"distilgpt2": {"enabled": True},        # Will download (328MB)
"gpt2": {"enabled": False},             # Won't download (548MB)
"microsoft/DialoGPT-large": {"enabled": False}  # Won't download (1.1GB)
```

## 🚀 Advanced Features

### **RAG Pipeline Components:**

1. **Document Processing:**
   - PyMuPDF for robust PDF text extraction
   - Text cleaning and normalization
   - Intelligent segmentation with overlap preservation

2. **Vector Operations:**
   - SentenceTransformers for dense embeddings
   - FAISS for efficient similarity search  
   - L2-normalized vectors for cosine similarity

3. **Generation Pipeline:**
   - Context-aware prompt engineering
   - Multiple generation backends with graceful fallback
   - Response quality filtering and validation

### **Batch Processing:**
```python
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = []

for doc in documents:
    qa.load_document(doc)
    answer = qa.ask_question("What is the main topic?")
    results.append((doc, answer))
```

## 🛡️ Troubleshooting

### **Model Issues**
**"No models enabled in configuration"**
- Edit `available_models` in `pdf_rag_qa.py`
- Set `"enabled": True` for desired models
- Save and restart the application

**"Model failed to load"**
- Check available RAM (need 2-4GB free)
- Try smaller model: `model distilgpt2`
- Install dependencies: `pip install transformers accelerate`

### **RAG Pipeline Issues**
**"No relevant segments found"**
- Document may not contain information about the query
- Try rephrasing the question
- Check if PDF text extraction was successful

**"Vector index build failed"**
- Ensure sentence-transformers is installed
- Check available memory (need 2GB+ free)
- Verify PDF contains extractable text

## 📈 RAG System Capabilities

**What the System Provides:**
✅ Complete RAG architecture with vector search  
✅ Semantic similarity matching with confidence scores  
✅ Context-aware answer generation  
✅ Multiple AI generation backends with fallback  
✅ Real-time model switching during conversations  
✅ Source attribution with page numbers and previews  
✅ Offline operation after initial setup  
✅ Production-ready error handling and logging  
✅ Intelligent model control (no unwanted downloads)

## ⚠️ Important Notes

- **Model Control**: Only models with `enabled: True` will be downloaded
- **First Run**: Enabled models download automatically (5-15 minutes)
- **Memory Usage**: Large models require 2-4GB RAM each
- **Privacy Mode**: Local-only processing available (`mode local`)
- **API Costs**: OpenAI integration optional (~$0.0005/question)

## 🎓 Perfect For

- **ML Engineers** - Complete RAG architecture reference
- **Data Scientists** - Vector search and embedding examples  
- **Developers** - Multi-model AI integration patterns
- **Researchers** - Local AI models for private analysis
- **Students** - Learn modern RAG system design

---

**Enterprise-Grade RAG Implementation**  
*Production-ready Retrieval-Augmented Generation system with intelligent model management*

**Ready for production use with enterprise-level features and reliability!** 🚀
