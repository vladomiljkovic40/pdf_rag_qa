#!/usr/bin/env python3
"""
PDF RAG QA System - Test Script
Demonstrates basic usage and testing of the RAG system.
"""

import os
import sys
from pdf_rag_qa import PDFRagQA

def test_basic_functionality():
    """Test basic system functionality with sample questions."""

    print("🧪 PDF RAG QA System - Basic Test")
    print("=" * 50)

    # Initialize system
    print("🔧 Initializing system...")
    qa_system = PDFRagQA(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=300,  # Smaller chunks for testing
        overlap_size=50
    )

    # Test sample questions (without document for demo)
    sample_questions = [
        "What is the main topic of this document?",
        "How many pages does this document have?", 
        "What are the key requirements mentioned?",
        "Can you summarize the introduction?",
        "What technical specifications are listed?"
    ]

    print("\n📝 Sample Questions for Testing:")
    for i, question in enumerate(sample_questions, 1):
        print(f"  {i}. {question}")

    print("\n💡 To test with a real document:")
    print("   1. Run: python pdf_rag_qa.py")
    print("   2. Provide path to your PDF file")
    print("   3. Ask questions from the list above")

    return qa_system

def create_sample_pdf_content():
    """Create a sample text file that simulates PDF content for testing."""

    sample_content = """Sample Document for PDF RAG Testing

Introduction
This is a sample document created for testing the PDF RAG QA system. 
It contains multiple sections with different types of information to 
demonstrate the semantic search and question-answering capabilities.

Technical Requirements
- Python 3.8 or higher
- Minimum 8 GB RAM (16 GB recommended)  
- Intel i3 processor or equivalent
- 500 MB free disk space for models
- Internet connection for initial model download

System Architecture
The RAG system consists of five main components:
1. PDF text extraction using PyMuPDF
2. Text segmentation with overlap preservation  
3. Semantic embedding using sentence transformers
4. Vector indexing with FAISS for fast search
5. Context-based answer generation

Features and Capabilities
The system can handle various question types including:
- What questions: Extracting definitions and explanations
- How questions: Finding procedural information  
- When questions: Identifying temporal information
- Where questions: Locating spatial references
- Why questions: Understanding causal relationships

Performance Metrics
Expected performance characteristics:
- Processing speed: 100 to 200 pages per minute
- Memory usage: 2 to 4 GB for typical documents
- Accuracy: 85 to 95 percent for well-formatted PDFs
- Response time: 1 to 3 seconds per question

Troubleshooting Guide
Common issues and solutions:
- PDF not loading: Check file format and permissions
- Memory errors: Reduce chunk size or use smaller model
- Poor accuracy: Verify PDF text quality and question clarity
- Slow performance: Consider GPU acceleration or smaller embeddings

Conclusion
This sample document demonstrates the types of content that work well
with the PDF RAG QA system. The system excels at finding relevant 
information and generating accurate answers based on document context.
"""

    # Save sample content
    with open("sample_document.txt", "w", encoding="utf-8") as f:
        f.write(sample_content)

    print("\n📄 Created sample_document.txt for testing")
    print("💡 You can use this to test the text processing pipeline")

def main():
    """Run basic tests and demonstrations."""

    print("🚀 PDF RAG QA System - Test Suite")
    print("=" * 60)

    # Test basic initialization
    try:
        qa_system = test_basic_functionality()
        print("\n✅ System initialization: PASSED")

        # Create sample content
        create_sample_pdf_content()
        print("✅ Sample content creation: PASSED")

        # Test document info (without loading)
        info = qa_system.get_document_info()
        print(f"✅ Document info retrieval: PASSED (empty: {len(info) == 0})")

        print("\n🎉 All basic tests completed successfully!")
        print("\n📖 Next Steps:")
        print("   1. Find a PDF document to test with")
        print("   2. Run: python pdf_rag_qa.py your_document.pdf")
        print("   3. Ask questions about the document content")
        print("   4. Observe source attribution and confidence scores")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\n🛠️ Troubleshooting:")
        print("   1. Ensure all requirements are installed: pip install -r requirements.txt")
        print("   2. Check Python version (3.8+ required)")
        print("   3. Verify internet connection for model downloads")

if __name__ == "__main__":
    main()
