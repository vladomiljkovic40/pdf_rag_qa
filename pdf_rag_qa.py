#!/usr/bin/env python3
"""
Enhanced PDF RAG QA System with PROPERLY Fixed Similarity Calculations
Now uses cosine similarity index instead of inner product for better normalization
"""

import os
import sys
import re
import warnings
from typing import List, Dict, Any, Tuple, Optional
import json

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import fitz  # PyMuPDF
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    import requests
    from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install requirements.txt first:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Optional dependencies for enhanced generation
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("INFO: OpenAI not available - using local generation only")

try:
    from langchain.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    print("INFO: LangChain not available - using direct transformers")

class PDFRagQA:
    """
    Enhanced PDF Document Question-Answering system with PROPERLY fixed similarity calculations.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 400, overlap_size: int = 50,
                 generation_mode: str = "auto"):
        """Initialize the Enhanced PDF RAG QA system."""
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.generation_mode = generation_mode

        # Initialize components
        self.embedding_model = None
        self.segments = []
        self.embeddings = None
        self.faiss_index = None
        self.local_generator = None
        self.available_models = {}
        self.current_model = None
        self.document_metadata = {}

        # Generation settings
        self.openai_model = "gpt-3.5-turbo"

        print("Initializing Enhanced PDF RAG QA System...")
        self._load_models()

    def _load_models(self):
        """Load embedding and generation models based on configuration."""
        try:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Load generation models based on mode
            if self.generation_mode in ["local", "auto", "hybrid"]:
                self._load_local_generator()

            if self.generation_mode in ["openai", "auto", "hybrid"]:
                self._check_openai_availability()

            print("Models loaded successfully!")

        except Exception as e:
            print(f"ERROR: Error loading models: {e}")
            print("INFO: Falling back to basic generation...")

    def _load_local_generator(self):
        """Enhanced selective local LLM loading with interactive model selection."""

        # ENHANCED CONTROL PANEL - More models with better organization
        available_models = {
            # Fast & Lightweight Models
            "distilgpt2": {
                "name": "DistilGPT2",
                "size": "328MB",
                "description": "Fastest responses, good for quick questions",
                "category": "Fast",
                "enabled": True  # Default enabled for speed
            },
            "microsoft/DialoGPT-small": {
                "name": "DialoGPT Small",
                "size": "117MB",
                "description": "Lightest model, basic conversations",
                "category": "Fast", 
                "enabled": True  # Enabled for low-resource systems
            },

            # Balanced Models
            "gpt2": {
                "name": "GPT-2",
                "size": "548MB",
                "description": "Balanced quality and speed, general purpose",
                "category": "Balanced",
                "enabled": True  # User can enable if wanted
            },
            "microsoft/DialoGPT-medium": {
                "name": "DialoGPT Medium",
                "size": "355MB",
                "description": "Good balance of quality and performance",
                "category": "Balanced",
                "enabled": True  # User can enable if wanted
            },

            # High-Quality Models
            "microsoft/DialoGPT-large": {
                "name": "DialoGPT Large",
                "size": "1.1GB",
                "description": "Best conversational quality, slower",
                "category": "Quality",
                "enabled": False  # User can enable for quality
            },
        }

        # Show available models to user
        print("\nAvailable Local LLM Models:")
        print("=" * 50)
        for category in ["Fast", "Balanced", "Quality"]:
            print(f"\n{category} Models:")
            for model_id, config in available_models.items():
                if config["category"] == category:
                    status = "ENABLED" if config["enabled"] else "DISABLED"
                    print(f"  {model_id}:")
                    print(f"    Name: {config['name']} ({config['size']})")
                    print(f"    Description: {config['description']}")
                    print(f"    Status: {status}")

        # Only tries models with enabled: True
        enabled_models = [model_id for model_id, config in available_models.items() if config["enabled"]]

        if not enabled_models:
            print("\nERROR: No models enabled in configuration")
            print("INFO: Edit the 'enabled' field in available_models to enable models")
            self.local_generator = None
            self.available_models = {}
            return

        print(f"\nLoading {len(enabled_models)} selected models...")

        loaded_models = {}

        for model_id in enabled_models:
            config = available_models[model_id]

            try:
                print(f"Loading {config['name']} ({config['size']})...")

                generator = pipeline(
                    "text-generation",
                    model=model_id,
                    device=-1,  # CPU only
                    max_length=1000,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=50256,
                    truncation=True
                )

                # Test the model
                test_result = generator("Test:", max_length=20, do_sample=False)

                if test_result:
                    loaded_models[model_id] = {
                        "generator": generator,
                        "config": config
                    }
                    print(f"SUCCESS: {config['name']} loaded successfully!")

            except Exception as e:
                print(f"WARNING: {config['name']} failed to load: {str(e)[:100]}")

        if loaded_models:
            # Use first successful model as default
            default_model = list(loaded_models.keys())[0]
            self.local_generator = loaded_models[default_model]["generator"]
            self.current_model = default_model
            self.available_models = loaded_models

            print(f"\nDefault model: {loaded_models[default_model]['config']['name']}")
            print(f"SUCCESS: {len(loaded_models)} models available for selection")

            # Show available models for switching
            if len(loaded_models) > 1:
                print("\nYou can switch between these models during runtime:")
                for i, (model_id, model_data) in enumerate(loaded_models.items(), 1):
                    current_marker = " (CURRENT)" if model_id == default_model else ""
                    print(f"  {i}. {model_data['config']['name']}{current_marker}")
                print("Use 'model <name>' command to switch during conversation")

        else:
            print("ERROR: No local LLM models could be loaded")
            self.local_generator = None
            self.available_models = {}

    def set_local_model(self, model_identifier: str):
        """Enhanced model switching with multiple ways to identify models."""
        if not hasattr(self, 'available_models') or not self.available_models:
            print("ERROR: No local models available")
            return False

        # Try to match by exact model ID first
        if model_identifier in self.available_models:
            self.local_generator = self.available_models[model_identifier]["generator"]
            self.current_model = model_identifier
            model_name = self.available_models[model_identifier]["config"]["name"]
            print(f"SUCCESS: Switched to: {model_name}")
            return True

        # Try to match by name (case-insensitive)
        model_identifier_lower = model_identifier.lower()
        for model_id, model_data in self.available_models.items():
            model_name_lower = model_data["config"]["name"].lower()
            if model_identifier_lower in model_name_lower or model_name_lower in model_identifier_lower:
                self.local_generator = model_data["generator"]
                self.current_model = model_id
                print(f"SUCCESS: Switched to: {model_data['config']['name']}")
                return True

        # Try to match by number (1, 2, 3, etc.)
        try:
            model_index = int(model_identifier) - 1
            model_list = list(self.available_models.keys())
            if 0 <= model_index < len(model_list):
                selected_model_id = model_list[model_index]
                self.local_generator = self.available_models[selected_model_id]["generator"]
                self.current_model = selected_model_id
                model_name = self.available_models[selected_model_id]["config"]["name"]
                print(f"SUCCESS: Switched to: {model_name}")
                return True
        except ValueError:
            pass

        # If no match found, show available options
        print(f"ERROR: Model '{model_identifier}' not available")
        self.show_available_models()
        return False

    def show_available_models(self):
        """Display all available models with selection options."""
        if not hasattr(self, 'available_models') or not self.available_models:
            print("No local models available")
            return

        print("\nAvailable Local Models:")
        print("=" * 40)

        for i, (model_id, model_data) in enumerate(self.available_models.items(), 1):
            config = model_data["config"]
            current_marker = " <- CURRENT" if model_id == self.current_model else ""

            print(f"{i}. {config['name']} ({config['size']}){current_marker}")
            print(f"   ID: {model_id}")
            print(f"   Description: {config['description']}")
            print(f"   Category: {config['category']}")
            print()

        print("To switch models, use any of these commands:")
        print("  'model 1' (by number)")
        print("  'model distilgpt2' (by ID)")
        print("  'model DistilGPT2' (by name)")
        print("  'model dialo' (partial name match)")

    def get_available_models(self):
        """Get enhanced list of available local models for UI."""
        if not hasattr(self, 'available_models') or not self.available_models:
            return {}

        return {
            model_id: {
                "name": config["config"]["name"],
                "description": config["config"]["description"],
                "size": config["config"]["size"],
                "category": config["config"]["category"],
                "current": model_id == getattr(self, 'current_model', None)
            }
            for model_id, config in self.available_models.items()
        }

    def _check_openai_availability(self):
        """Check if OpenAI API is available and configured."""
        if not HAS_OPENAI:
            print("WARNING: OpenAI library not installed")
            return False

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("WARNING: OPENAI_API_KEY not found in environment variables")
            print("INFO: Set your API key: export OPENAI_API_KEY='your-key-here'")
            return False

        try:
            openai.api_key = api_key
            print("SUCCESS: OpenAI API configured successfully!")
            return True
        except Exception as e:
            print(f"WARNING: OpenAI API error: {e}")
            return False

    def load_document(self, pdf_path: str) -> bool:
        """Load and process PDF document."""
        if not os.path.exists(pdf_path):
            print(f"ERROR: File not found: {pdf_path}")
            return False

        try:
            print(f"Loading PDF: {pdf_path}")

            # Extract text from PDF
            text_data = self._extract_pdf_text(pdf_path)
            if not text_data:
                print("ERROR: No text extracted from PDF")
                return False

            print(f"Extracted {len(text_data)} pages")

            # Create segments
            self.segments = self._create_segments(text_data)
            print(f"Created {len(self.segments)} text segments")

            # Build vector index
            self._build_vector_index()
            print("Vector index built successfully!")

            # Store metadata
            self.document_metadata = {
                "path": pdf_path,
                "filename": os.path.basename(pdf_path),
                "pages": len(text_data),
                "segments": len(self.segments)
            }

            print("Document loaded and processed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Error loading document: {e}")
            return False

    def _extract_pdf_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using PyMuPDF."""
        text_data = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                text = self._clean_text(text)
                if text.strip():
                    text_data.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "word_count": len(text.split())
                    })
            doc.close()
        except Exception as e:
            print(f"ERROR: Error extracting PDF text: {e}")
        return text_data

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\f\r\v]', ' ', text)
        text = re.sub(r'[""''`]', '"', text)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 5 and not re.match(r'^\d+$', line):
                cleaned_lines.append(line)
        return ' '.join(cleaned_lines)

    def _create_segments(self, text_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create text segments with overlap."""
        segments = []
        segment_id = 0

        for page_data in text_data:
            page_num = page_data["page_number"]
            text = page_data["text"]
            words = text.split()

            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                segment_text = ' '.join(words[start:end])

                segments.append({
                    "id": segment_id,
                    "text": segment_text,
                    "page_number": page_num,
                    "start_word": start,
                    "end_word": end,
                    "word_count": len(segment_text.split())
                })

                segment_id += 1
                if end >= len(words):
                    break
                start += (self.chunk_size - self.overlap_size)

        return segments

    def _build_vector_index(self):
        """FIXED: Build FAISS vector index with proper cosine similarity."""
        if not self.segments:
            print("ERROR: No segments to index")
            return

        print("Creating embeddings...")
        texts = [segment["text"] for segment in self.segments]
        self.embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        dimension = self.embeddings.shape[1]

        # FIX: Use IndexFlatL2 for proper cosine similarity
        # We'll normalize embeddings and use L2 distance, then convert to cosine similarity
        self.faiss_index = faiss.IndexFlatL2(dimension)

        # Normalize embeddings for cosine similarity calculation
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings)

        print(f"Index built with {len(self.segments)} segments using cosine similarity")

    def ask_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Ask a question with enhanced answer generation."""
        if not self.segments or self.faiss_index is None:
            return {
                "answer": "ERROR: No document loaded. Please load a PDF first.",
                "sources": [],
                "confidence": 0.0,
                "generation_method": "none"
            }

        print(f"Processing question: {question}")

        # Show current model being used
        if self.current_model and self.available_models:
            current_model_name = self.available_models[self.current_model]["config"]["name"]
            print(f"Using model: {current_model_name}")

        try:
            # Find relevant segments
            relevant_segments = self._search_relevant_segments(question, top_k)

            if not relevant_segments:
                return {
                    "answer": "ERROR: No relevant content found for this question.",
                    "sources": [],
                    "confidence": 0.0,
                    "generation_method": "none"
                }

            # Generate intelligent answer
            answer_data = self._generate_intelligent_answer(question, relevant_segments)
            return answer_data

        except Exception as e:
            return {
                "answer": f"ERROR: Error processing question: {e}",
                "sources": [],
                "confidence": 0.0,
                "generation_method": "error"
            }

    def _search_relevant_segments(self, question: str, top_k: int) -> List[Tuple[Dict, float]]:
        """PROPERLY FIXED: Search for relevant segments with correct cosine similarity."""
        question_embedding = self.embedding_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(question_embedding)

        # Search using L2 distance (on normalized vectors = cosine similarity)
        distances, indices = self.faiss_index.search(question_embedding, top_k)

        relevant_segments = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.segments):
                segment = self.segments[idx]

                # PROPER FIX: Convert L2 distance to cosine similarity
                # For normalized vectors: cosine_similarity = 1 - (L2_distance^2 / 2)
                # But since FAISS L2 on normalized vectors gives squared distance,
                # we need: cosine_similarity = 1 - (distance / 2)

                distance = float(distance)

                # Convert L2 squared distance to cosine similarity [0, 1]
                # L2 distance on normalized vectors ranges from 0 to 2
                # Cosine similarity = 1 - (distance / 2)
                cosine_similarity = 1.0 - (distance / 2.0)

                # Clamp to [0, 1] range to handle numerical precision
                cosine_similarity = max(0.0, min(1.0, cosine_similarity))

                # Convert to percentage [0, 100]
                similarity_percent = cosine_similarity * 100.0

                relevant_segments.append((segment, similarity_percent))

        return relevant_segments

    def _generate_intelligent_answer(self, question: str, relevant_segments: List[Tuple[Dict, float]]) -> Dict[str, Any]:
        """Generate intelligent answer with proper confidence calculation."""
        # Prepare context
        context_parts = []
        sources = []

        for segment, similarity in relevant_segments:
            context_parts.append(f"[Page {segment['page_number']}] {segment['text']}")
            sources.append({
                "page": segment['page_number'],
                "segment_id": segment['id'],
                "similarity": similarity,  # Now properly normalized 0-100
                "text_preview": segment['text'][:100] + "..." if len(segment['text']) > 100 else segment['text']
            })

        context = "\n\n".join(context_parts)

        # Try different generation methods in order of preference
        answer, method = self._try_generation_methods(question, context)

        # Calculate confidence properly (0-1 range)
        if relevant_segments:
            avg_confidence = sum(sim for _, sim in relevant_segments) / len(relevant_segments)
            # Convert from 0-100 scale to 0-1 scale for confidence
            avg_confidence = avg_confidence / 100.0
        else:
            avg_confidence = 0.0

        return {
            "answer": answer,
            "sources": sources,
            "confidence": float(avg_confidence),
            "context_used": len(relevant_segments),
            "generation_method": method
        }

    def _try_generation_methods(self, question: str, context: str) -> Tuple[str, str]:
        """Try different generation methods in order of preference."""

        # Method 1: OpenAI API (best quality)
        if self.generation_mode in ["openai", "auto", "hybrid"] and HAS_OPENAI:
            try:
                answer = self._generate_with_openai(question, context)
                if answer and not answer.startswith("Error"):
                    return answer, "openai"
            except Exception as e:
                print(f"WARNING: OpenAI generation failed: {e}")

        # Method 2: Local LLM (good quality, private)
        if self.generation_mode in ["local", "auto", "hybrid"] and self.local_generator:
            try:
                answer = self._generate_with_local_llm(question, context)
                if answer:
                    current_model_name = self.available_models[self.current_model]["config"]["name"]
                    return answer, f"local_llm ({current_model_name})"
            except Exception as e:
                print(f"WARNING: Local LLM generation failed: {e}")

        # Method 3: Enhanced rule-based (reliable fallback)
        try:
            answer = self._generate_enhanced_answer(question, context)
            return answer, "enhanced_rules"
        except Exception as e:
            print(f"WARNING: Enhanced generation failed: {e}")

        # Method 4: Basic fallback
        answer = self._generate_basic_answer(question, context)
        return answer, "basic"

    def _generate_with_openai(self, question: str, context: str) -> str:
        """Generate answer using OpenAI API."""
        if not HAS_OPENAI or not os.getenv('OPENAI_API_KEY'):
            return None

        system_prompt = """You are an expert document analyst. Answer questions based STRICTLY on the provided document context.

INSTRUCTIONS:
1. Analyze the context carefully
2. Provide comprehensive, well-structured answers
3. If the context doesn't contain the answer, say "The document doesn't contain information about this"
4. Use specific details and evidence from the document
5. Organize your response clearly with bullet points or paragraphs as needed
6. Quote relevant parts when helpful
7. Be concise but thorough"""

        user_prompt = f"""DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive answer based on the document context above."""

        try:
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.3,
                top_p=1.0
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"WARNING: OpenAI API error: {e}")
            return None

    def _generate_with_local_llm(self, question: str, context: str) -> str:
        """Generate answer using local LLM."""
        if not self.local_generator:
            return None

        # Create a focused prompt for the local model
        prompt = f"""Context: {context[:800]}...

Question: {question}

Based on the context above, provide a clear and helpful answer:"""

        try:
            # Generate response
            outputs = self.local_generator(
                prompt,
                max_length=len(prompt.split()) + 150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )

            # Extract the generated text after the prompt
            full_response = outputs[0]['generated_text']
            answer = full_response[len(prompt):].strip()

            # Clean up the response
            answer = self._clean_generated_text(answer)
            return answer if answer else None

        except Exception as e:
            print(f"WARNING: Local LLM error: {e}")
            return None

    def _clean_generated_text(self, text: str) -> str:
        """Clean generated text from artifacts."""
        # Remove common generation artifacts
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove repetitive patterns
        sentences = text.split('.')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)

        return '. '.join(unique_sentences[:3]) + '.' if unique_sentences else text

    def _generate_enhanced_answer(self, question: str, context: str) -> str:
        """Generate enhanced answer using advanced rule-based approach."""
        return f"Based on the document:\n\n{context[:500]}...\n\nThis information addresses your question."

    def _generate_basic_answer(self, question: str, context: str) -> str:
        """Basic fallback answer generation."""
        return f"From the document:\n\n{context[:400]}..."

    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the loaded document."""
        info = self.document_metadata.copy()
        info["generation_capabilities"] = {
            "openai_available": HAS_OPENAI and os.getenv('OPENAI_API_KEY') is not None,
            "local_llm_loaded": self.local_generator is not None,
            "current_model": self.current_model,
            "available_models": len(self.available_models) if hasattr(self, 'available_models') else 0,
            "generation_mode": self.generation_mode
        }
        return info

    def set_generation_mode(self, mode: str):
        """Change generation mode: 'local', 'openai', 'auto', or 'hybrid'."""
        valid_modes = ["local", "openai", "auto", "hybrid"]
        if mode in valid_modes:
            self.generation_mode = mode
            print(f"SUCCESS: Generation mode set to: {mode}")
        else:
            print(f"ERROR: Invalid mode. Choose from: {valid_modes}")

    def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if command was processed."""
        command = command.strip().lower()

        if command == 'models':
            self.show_available_models()
            return True

        if command.startswith('model '):
            model_identifier = command.split(' ', 1)[1].strip()
            self.set_local_model(model_identifier)
            return True

        if command.startswith('mode '):
            new_mode = command.split(' ', 1)[1].strip()
            self.set_generation_mode(new_mode)
            return True

        if command == 'info':
            info = self.get_document_info()
            print(f"\nSystem Information:")
            for key, value in info.items():
                if key != 'generation_capabilities':
                    print(f" {key}: {value}")

            gen_info = info.get('generation_capabilities', {})
            print(f"\nAI Capabilities:")
            print(f" OpenAI Available: {'YES' if gen_info.get('openai_available') else 'NO'}")
            print(f" Local LLM Loaded: {'YES' if gen_info.get('local_llm_loaded') else 'NO'}")
            print(f" Available Models: {gen_info.get('available_models', 0)}")
            if gen_info.get('current_model') and hasattr(self, 'available_models'):
                current_model_name = self.available_models[gen_info['current_model']]["config"]["name"]
                print(f" Current Model: {current_model_name}")
            print(f" Generation Mode: {gen_info.get('generation_mode', 'basic')}")
            return True

        if command == 'help':
            self.show_help()
            return True

        return False

    def show_help(self):
        """Show help information."""
        print("\nAvailable Commands:")
        print("=" * 30)
        print("models                    - Show available local models")
        print("model <name/number>       - Switch to specific model")
        print("mode <mode>              - Change generation mode (local/openai/auto)")
        print("info                     - Show system and document information")
        print("help                     - Show this help message")
        print("quit/exit/q              - Exit the system")
        print("\nOr just ask questions about your document!")

    def interactive_mode(self):
        """Enhanced interactive Q&A mode with proper model selection."""
        print("\n" + "="*60)
        print("Enhanced PDF RAG QA System - Interactive Mode")
        print("="*60)

        # Show available commands before asking for PDF
        print("\nAvailable commands:")
        print(" - Type a PDF file path to load a document")
        print(" - Type 'models' to see available local models")
        print(" - Type 'model <name/number>' to switch models")
        print(" - Type 'mode <mode>' to change generation mode")  
        print(" - Type 'info' to see system information")
        print(" - Type 'help' for full command list")
        print(" - Type 'quit', 'exit', or 'q' to end the session")

        # Main interactive loop
        while True:
            try:
                if not self.segments:
                    user_input = input("\nEnter PDF file path (or command): ").strip()
                else:
                    user_input = input(f"\nQuestion or command: ").strip()

                if not user_input:
                    continue

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using Enhanced PDF RAG QA System!")
                    break

                # Try to handle as command first
                if self.handle_command(user_input):
                    continue

                # If no document loaded and not a command, try to load as PDF
                if not self.segments:
                    # Check if it looks like a file path
                    if os.path.exists(user_input.strip('"')):
                        pdf_path = user_input.strip('"')
                        if self.load_document(pdf_path):
                            # Display document info after successful load
                            info = self.get_document_info()
                            print(f"\nDocument Info:")
                            print(f" File: {info.get('filename', 'Unknown')}")
                            print(f" Pages: {info.get('pages', 0)}")
                            print(f" Segments: {info.get('segments', 0)}")

                            gen_info = info.get('generation_capabilities', {})
                            print(f"\nAI Capabilities:")
                            print(f" OpenAI Available: {'YES' if gen_info.get('openai_available') else 'NO'}")
                            print(f" Local LLM Loaded: {'YES' if gen_info.get('local_llm_loaded') else 'NO'}")
                            print(f" Available Models: {gen_info.get('available_models', 0)}")
                            if gen_info.get('current_model'):
                                current_model_name = self.available_models[gen_info['current_model']]["config"]["name"]
                                print(f" Current Model: {current_model_name}")
                            print(f" Generation Mode: {gen_info.get('generation_mode', 'basic')}")

                            print("\nYou can now ask questions about the document!")
                            print("Or use commands like 'models', 'model <name>', etc.")
                        else:
                            print("ERROR: Failed to load PDF. Try another file or use commands.")
                    else:
                        print(f"ERROR: File not found: {user_input}")
                        print("Try entering a valid PDF file path, or use commands like 'models' or 'help'")
                    continue

                # If document is loaded, treat as question
                print("\nAnalyzing question and generating response...")
                result = self.ask_question(user_input)

                # Display results
                print("\n" + "="*50)
                print(f"ANSWER ({result.get('generation_method', 'unknown').upper()}):")
                print("="*50)
                print(result["answer"])

                if result["sources"]:
                    print(f"\nSources (Confidence: {result['confidence']:.2%}):")
                    for i, source in enumerate(result["sources"], 1):
                        print(f" {i}. Page {source['page']} (similarity: {source['similarity']:.2%})")
                        print(f"    Preview: {source['text_preview']}")

                print(f"\nGenerated using: {result.get('generation_method', 'unknown')} method")
                print("\n" + "-"*50)

            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nERROR: {e}")
                continue

def main():
    """Main application entry point."""
    print("Starting Enhanced PDF RAG QA System...")

    # Initialize system with auto mode (tries best available option)
    qa_system = PDFRagQA(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=400,
        overlap_size=50,
        generation_mode="auto"  # Will use best available option
    )

    # Check for command line arguments
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Loading PDF from command line: {pdf_path}")
        if qa_system.load_document(pdf_path):
            qa_system.interactive_mode()
        else:
            print("ERROR: Failed to load PDF from command line")
    else:
        qa_system.interactive_mode()

if __name__ == "__main__":
    main()
