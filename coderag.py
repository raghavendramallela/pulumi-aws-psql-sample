"""
CODE RAG (Retrieval-Augmented Generation) System
=================================================
This script implements a complete RAG pipeline for code repositories:
1. Clones a Git repository
2. Parses and chunks code files
3. Creates vector embeddings for semantic search
4. Uses an LLM to answer questions about the code

Dependencies (install with uv pip):
"""
# uv pip install ipykernal
# uv pip install -q gitpython==3.1.46
# uv pip install -q tree-sitter==0.25.2 tree-sitter-languages==1.10.2
# uv pip install -q sentence-transformers==5.2.2
# uv pip install -q faiss-cpu==1.13.2
# uv pip install -q transformers==5.0.0 accelerate==1.12.0 bitsandbytes==0.49.1
# uv pip install -q torch>=2.3.0

import sys
# Check Python version - need 3.10+ for best compatibility
if sys.version_info < (3, 10):
    print("⚠️  Warning: Python 3.10+ recommended. Current:", sys.version)

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import subprocess

# Git operations for cloning repositories
import git
# Tree-sitter for parsing code into AST (Abstract Syntax Tree)
from tree_sitter_languages import get_parser, get_language
# NumPy for numerical operations
import numpy as np
# FAISS for efficient similarity search in vector space
import faiss
# Sentence transformers for creating embeddings
from sentence_transformers import SentenceTransformer
# PyTorch for deep learning operations
import torch
# Hugging Face transformers for loading LLMs
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Verify PyTorch version for compatibility
print(f"PyTorch version: {torch.__version__}")
if torch.__version__ < "2.1.0":
    print("⚠️  Warning: PyTorch 2.3+ recommended for transformers v5.0")

# Check available GPUs
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"\n🎮 Detected {gpu_count} GPU(s):")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()


class GitRepoIngestion:
    """
    STEP 1: Clone and extract code files from Git repository
    
    This class handles:
    - Cloning a Git repository to local disk
    - Walking through the directory structure
    - Extracting all code files with supported extensions
    """

    def __init__(self, repo_url: str, local_path: str = "./repo"):
        """
        Initialize the repository ingestion system
        
        Args:
            repo_url: GitHub/GitLab URL to clone
            local_path: Where to store the cloned repo locally
        """
        self.repo_url = repo_url
        self.local_path = local_path
        # Define which file extensions we consider "code"
        self.code_extensions = {
            '.js', '.ts', '.tsx', '.jsx', '.py', '.java', '.go', '.rb',
            '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.sh', '.bash',
            '.yml', '.yaml', '.json', '.md', '.rs', '.swift', '.kt'
        }

    def clone_repo(self):
        """
        Clone the Git repository to local disk
        
        Returns:
            git.Repo object representing the cloned repository
        """
        # Check if repo already exists to avoid re-cloning
        if os.path.exists(self.local_path):
            print(f"Repository already exists at {self.local_path}")
            return git.Repo(self.local_path)

        # Clone the repository from the URL
        print(f"Cloning {self.repo_url}...")
        repo = git.Repo.clone_from(self.repo_url, self.local_path)
        print(f"✓ Cloned successfully")
        return repo

    def get_code_files(self) -> List[Dict[str, str]]:
        """
        Walk through the repository and extract all code files
        
        Returns:
            List of dictionaries containing file path, content, and language
        """
        files = []
        # Directories to skip (dependencies, build artifacts, caches)
        exclude_dirs = {
            '.git', 'node_modules', '__pycache__', 'dist', 'build',
            '.venv', 'venv', '.pytest_cache', '.mypy_cache', 'target',
            'bin', 'obj', '.gradle'
        }

        # Walk through all directories in the repository
        for root, dirs, filenames in os.walk(self.local_path):
            # Skip excluded directories (modifies dirs in-place)
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for filename in filenames:
                file_path = Path(root) / filename
                # Only process files with code extensions
                if file_path.suffix in self.code_extensions:
                    try:
                        # Read file content with UTF-8 encoding
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        # Skip very large files (>500KB) to avoid memory issues
                        if len(content) > 500_000:
                            continue

                        # Store relative path (not absolute)
                        relative_path = file_path.relative_to(self.local_path)
                        files.append({
                            'path': str(relative_path),
                            'content': content,
                            'language': file_path.suffix[1:]  # Remove the dot from extension
                        })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        print(f"✓ Found {len(files)} code files")
        return files


class CodeChunker:
    """
    STEP 2: Parse and chunk code into meaningful segments
    
    This class:
    - Uses tree-sitter to parse code into AST (Abstract Syntax Tree)
    - Extracts functions, classes, and methods as separate chunks
    - Falls back to simple line-based chunking for unsupported languages
    
    Why chunk? Large files need to be broken into smaller pieces for:
    - Better embedding quality (focused semantic meaning)
    - More precise retrieval (find specific functions, not whole files)
    """

    def __init__(self):
        """Initialize with empty parser cache"""
        # Cache parsers to avoid reloading them for each file
        self.parser_cache = {}

    def get_parser(self, language: str):
        """
        Get tree-sitter parser for a specific programming language
        
        Tree-sitter parsers convert code into an AST (Abstract Syntax Tree)
        which allows us to identify functions, classes, etc.
        
        Args:
            language: File extension (e.g., 'py', 'js', 'ts')
            
        Returns:
            Parser object or None if not available
        """
        # Map file extensions to tree-sitter language names
        lang_map = {
            'js': 'javascript',
            'jsx': 'javascript',
            'ts': 'typescript',
            'tsx': 'typescript',
            'py': 'python',
            'java': 'java',
            'go': 'go',
            'rb': 'ruby',
            'cpp': 'cpp',
            'c': 'c',
            'cs': 'c_sharp',
            'sh': 'bash',
            'bash': 'bash',
            'rs': 'rust',
            'php': 'php'
        }

        lang = lang_map.get(language, language)
        # Load parser only once and cache it
        if lang not in self.parser_cache:
            try:
                self.parser_cache[lang] = get_parser(lang)
            except Exception as e:
                print(f"Parser not available for {lang}: {e}")
                return None
        return self.parser_cache.get(lang)

    def chunk_code(self, files: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Create chunks from code files using intelligent parsing
        
        Strategy:
        1. Try tree-sitter parsing for supported languages (extracts functions/classes)
        2. Fall back to simple line-based chunking if parsing fails
        
        Args:
            files: List of file dictionaries from GitRepoIngestion
            
        Returns:
            List of chunk dictionaries with content, metadata, and location info
        """
        chunks = []

        for file in files:
            # Try to get a parser for this language
            parser = self.get_parser(file['language'])

            # Use tree-sitter for well-supported languages
            if parser and file['language'] in ['py', 'js', 'ts', 'jsx', 'tsx', 'java', 'go', 'rs']:
                file_chunks = self._parse_with_tree_sitter(file, parser)
                if file_chunks:
                    chunks.extend(file_chunks)
                else:
                    # Fallback if parser didn't extract anything
                    chunks.extend(self._simple_chunk(file))
            else:
                # Fallback: chunk by logical blocks (for YAML, JSON, Markdown, etc.)
                chunks.extend(self._simple_chunk(file))

        print(f"✓ Created {len(chunks)} code chunks")
        return chunks

    def _parse_with_tree_sitter(self, file: Dict, parser) -> List[Dict]:
        """
        Parse file using tree-sitter to extract functions and classes
        
        Tree-sitter creates an AST (Abstract Syntax Tree) which represents
        the structure of the code. We walk this tree to find meaningful chunks.
        
        Args:
            file: File dictionary with content and metadata
            parser: Tree-sitter parser for this language
            
        Returns:
            List of chunk dictionaries or empty list if parsing fails
        """
        chunks = []
        try:
            # Parse the file content into an AST
            tree = parser.parse(bytes(file['content'], 'utf8'))

            # Recursively extract function and class definitions from the AST
            self._extract_nodes(tree.root_node, file, chunks)

            # If no chunks extracted, return empty to trigger fallback
            if not chunks:
                return []

        except Exception as e:
            print(f"Parser error for {file['path']}: {e}")
            return []

        return chunks

    def _extract_nodes(self, node, file: Dict, chunks: List):
        """
        Recursively walk the AST and extract meaningful code blocks
        
        This function traverses the tree structure and identifies nodes that
        represent functions, classes, methods, etc.
        
        Args:
            node: Current AST node being examined
            file: File dictionary with content and metadata
            chunks: List to append extracted chunks to (modified in-place)
        """
        # Node types that represent meaningful code blocks
        # Different languages use different names for the same concepts
        interesting_types = {
            'function_definition', 'function_declaration', 'function_item',  # Python, JS, Rust
            'class_definition', 'class_declaration', 'class_item',  # Classes
            'method_definition', 'method_declaration',  # Methods
            'interface_declaration', 'struct_item',  # TypeScript, Rust
            'impl_item',  # Rust implementations
        }

        # If this node is a function/class/method, extract it
        if node.type in interesting_types:
            # Get the actual code text for this node
            code = file['content'][node.start_byte:node.end_byte]

            # Try to extract the name (e.g., function name, class name)
            name = self._extract_name(node, file['content'])

            # Create a chunk with all relevant metadata
            chunks.append({
                'content': code,
                'path': file['path'],
                'language': file['language'],
                'type': node.type,
                'name': name,
                'start_line': node.start_point[0] + 1,  # Convert 0-indexed to 1-indexed
                'end_line': node.end_point[0] + 1
            })

        # Recursively process all child nodes
        for child in node.children:
            self._extract_nodes(child, file, chunks)

    def _extract_name(self, node, content: str) -> str:
        """
        Extract the name of a function/class from an AST node
        
        Args:
            node: AST node representing a function or class
            content: Full file content
            
        Returns:
            Name string or "unknown" if not found
        """
        try:
            # Look for identifier in children (the name of the function/class)
            for child in node.children:
                if 'identifier' in child.type or child.type == 'name':
                    return content[child.start_byte:child.end_byte]
        except:
            pass
        return "unknown"

    def _simple_chunk(self, file: Dict) -> List[Dict]:
        """
        Fallback chunking strategy: split by line count with overlap
        
        Used when tree-sitter parsing isn't available or fails.
        Creates overlapping chunks to avoid cutting context at boundaries.
        
        Args:
            file: File dictionary with content and metadata
            
        Returns:
            List of chunk dictionaries
        """
        content = file['content']
        lines = content.split('\n')

        # Special handling for README files - keep them mostly whole
        # READMEs often have important context that shouldn't be split
        if 'readme' in file['path'].lower():
            chunk_size = 200  # Larger chunks for README
            overlap = 20
        else:
            chunk_size = 80  # ~80 lines per chunk
            overlap = 15     # 15 lines overlap between chunks

        chunks = []

        # Create overlapping chunks by stepping through lines
        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            # Skip empty chunks (all whitespace)
            if chunk_lines and any(line.strip() for line in chunk_lines):
                chunks.append({
                    'content': '\n'.join(chunk_lines),
                    'path': file['path'],
                    'language': file['language'],
                    'type': 'chunk',
                    'name': f'chunk_{i}',
                    'start_line': i + 1,
                    'end_line': min(i + len(chunk_lines), len(lines))
                })

        return chunks


class CodeVectorStore:
    """
    STEP 3: Embed code chunks and enable similarity search
    
    This class:
    - Converts code chunks into vector embeddings (numerical representations)
    - Stores embeddings in a FAISS index for fast similarity search
    - Enables semantic search: find code similar in meaning to a query
    
    How it works:
    1. Each chunk is converted to a vector (e.g., 384 dimensions)
    2. Similar code has similar vectors (close in vector space)
    3. FAISS finds nearest neighbors efficiently (even with millions of vectors)
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cuda:0"):
        """
        Initialize with an embedding model
        
        Embedding models convert text into vectors that capture semantic meaning.
        
        Options:
        - BAAI/bge-small-en-v1.5 (recommended, general purpose, better than MiniLM)
        - jinaai/jina-embeddings-v2-base-code (code-specific, 8192 context)
        - sentence-transformers/all-mpnet-base-v2 (good alternative)
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run embedding model on (default: cuda:0 - first GPU)
        """
        print(f"Loading embedding model: {model_name} on {device}...")
        self.embedding_model = SentenceTransformer(
            model_name,
            trust_remote_code=True,  # Required for some newer models
            device=device  # Place embedding model on first GPU
        )
        self.chunks = []  # Store original chunks for retrieval
        self.index = None  # FAISS index (created later)
        print(f"✓ Embedding model loaded on {device} (dim: {self.embedding_model.get_sentence_embedding_dimension()})")

    def embed_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Generate vector embeddings for all code chunks
        
        This is the core of the RAG system:
        1. Convert each chunk into a rich text representation
        2. Use the embedding model to convert text -> vector
        3. Store vectors in FAISS for fast similarity search
        
        Args:
            chunks: List of chunk dictionaries from CodeChunker
        """
        self.chunks = chunks  # Store original chunks for later retrieval

        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Create rich text representations for embedding
        # Include metadata to improve retrieval quality
        texts = []
        for c in chunks:
            # Build a descriptive header with file path and language
            metadata = f"File: {c['path']} | Language: {c['language']}"
            if c.get('name') and c['name'] != 'unknown':
                metadata += f" | {c['type']}: {c['name']}"

            # Combine metadata + code content (limit to 2000 chars to fit in model)
            text = f"{metadata}\n\n{c['content'][:2000]}"
            texts.append(text)

        # Generate embeddings using the sentence transformer model
        # Each text becomes a vector (e.g., 384 dimensions for BGE-small)
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,  # Process 32 chunks at a time
            normalize_embeddings=True  # Normalize vectors for cosine similarity
        )

        # Create FAISS index for efficient similarity search
        # IndexFlatIP = Inner Product (equivalent to cosine similarity with normalized vectors)
        dimension = embeddings.shape[1]  # Get embedding dimension (e.g., 384)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))  # Add all embeddings to index

        print(f"✓ Vector store created with {self.index.ntotal} embeddings")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for code chunks most relevant to a query
        
        How it works:
        1. Convert query text into a vector (same embedding model)
        2. Find vectors in FAISS index closest to query vector
        3. Return the original chunks corresponding to those vectors
        
        Args:
            query: Natural language question or search term
            top_k: Number of results to return
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        # Convert query into a vector using the same embedding model
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True  # Must normalize to match index
        )

        # Search FAISS index for nearest neighbors
        # Returns: scores (similarity values) and indices (positions in chunks list)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Build result list with original chunks + similarity scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):  # Safety check to avoid index errors
                chunk = self.chunks[idx].copy()  # Copy to avoid modifying original
                chunk['score'] = float(score)  # Add similarity score
                results.append(chunk)

        return results

class CodeQABot:
    """
    STEP 4: LLM-based Q&A bot for code
    
    This class:
    - Loads a large language model (LLM) specialized for code
    - Takes a question + retrieved code chunks as context
    - Generates natural language answers about the code
    
    Uses 4-bit quantization to reduce memory usage (14B model -> ~8GB RAM)
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct", use_multi_gpu: bool = True):
        """
        Initialize with a code-specialized LLM
        
        Qwen2.5-Coder is trained specifically on code and technical content.
        We use 4-bit quantization to fit large models in GPU memory.
        
        With multi-GPU setup (GPU-only, no CPU offloading):
        - Embedding model runs on GPU 0 (RTX 4080)
        - LLM is distributed across GPU 0 and GPU 1 (RTX 4080 + 5080)
        
        Recommended models for your setup (GPU-only):
        - Qwen/Qwen2.5-Coder-14B-Instruct (~8GB VRAM with 4-bit) ✓ Fits easily
        - Qwen/Qwen2.5-Coder-32B-Instruct (~18GB VRAM with 4-bit) ✓ Should fit
        - Qwen/Qwen3-Coder-30B-A3B-Instruct (~18-20GB VRAM with 4-bit) ✓ Should fit (tight)
        
        Args:
            model_name: Hugging Face model identifier
            use_multi_gpu: Whether to distribute model across multiple GPUs
        """
        print(f"Loading LLM: {model_name}...")
        
        # Check available GPUs
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if gpu_count < 2 and use_multi_gpu:
            print(f"  Warning: Only {gpu_count} GPU(s) available, using single GPU mode")
            use_multi_gpu = False
        
        # Configure 4-bit quantization to reduce memory usage
        # This compresses the model from 16-bit to 4-bit precision (~75% memory reduction)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,  # Compute in float16
            bnb_4bit_use_double_quant=True,  # Double quantization for better quality
            bnb_4bit_quant_type="nf4"  # NormalFloat4 quantization type
        )

        # Load tokenizer (converts text <-> tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True  # Required for some models
        )

        # Configure device mapping for multi-GPU setup (GPU-only, no CPU offloading)
        if use_multi_gpu and gpu_count >= 2:
            print(f"  Distributing model across {gpu_count} GPUs (GPU-only mode)...")
            
            # Calculate available memory per GPU
            # RTX 4080 Super: 16GB total, RTX 5080: 16GB total
            # Reserve memory for embedding model on GPU 0
            max_memory = {
                0: "13GB",  # GPU 0: Reserve 3GB for embedding model + overhead
                1: "15GB"   # GPU 1: Almost all memory available for LLM
                # No CPU entry = no CPU offloading
            }
            
            print(f"  Memory allocation: GPU 0: 13GB, GPU 1: 15GB (Total: 28GB for LLM)")
        else:
            print(f"  Using single GPU mode...")
            max_memory = None

        # Load the actual LLM with quantization and multi-GPU support
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",  # Distribute across GPUs automatically
                max_memory=max_memory,  # Memory limits per GPU (GPU-only)
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use half precision
                low_cpu_mem_usage=True  # Reduce CPU memory during loading
            )
        except ValueError as e:
            if "CPU or the disk" in str(e):
                print(f"\n  ❌ Error: Model too large for available GPU memory!")
                print(f"  The model requires more than the available ~28GB GPU memory.")
                print(f"\n  Solutions:")
                print(f"  1. Use a smaller model (recommended):")
                print(f"     - Qwen/Qwen2.5-Coder-14B-Instruct (~8GB)")
                print(f"     - Qwen/Qwen2.5-Coder-32B-Instruct (~18GB)")
                print(f"  2. Use 8-bit quantization instead of 4-bit (requires more VRAM)")
                print(f"  3. Enable CPU offloading (slower, but works)")
                raise
            else:
                raise

        # Set padding token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✓ LLM loaded (4-bit quantized, {model_name})")
        
        # Show memory usage per GPU
        if torch.cuda.is_available():
            total_allocated = 0
            for i in range(gpu_count):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                total_allocated += mem_allocated
                print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved")
            print(f"  Total GPU memory used: {total_allocated:.1f}GB")
        
        # Print device map to show layer distribution
        if hasattr(self.model, 'hf_device_map'):
            print(f"\n  Model layer distribution:")
            device_summary = {}
            for layer, device in self.model.hf_device_map.items():
                device_summary[device] = device_summary.get(device, 0) + 1
            for device, count in sorted(device_summary.items()):
                print(f"    {device}: {count} layers")

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate an answer to a question using retrieved code as context
        
        This is the "Generation" part of RAG:
        1. Build a prompt with the question + relevant code chunks
        2. Feed to LLM to generate an answer
        3. Clean up the response
        
        Args:
            query: User's question
            context_chunks: Retrieved code chunks from vector search
            
        Returns:
            Natural language answer as a string
        """

        # Build context section with code chunks
        context = "# CODE CONTEXT:\n\n"
        for i, chunk in enumerate(context_chunks[:3], 1):  # Use top 3 chunks
            # Create a descriptive header for each chunk
            header = f"## File: {chunk['path']} (Lines {chunk['start_line']}-{chunk['end_line']})"
            if chunk.get('name') and chunk['name'] != 'unknown':
                header += f" - {chunk['name']}"

            # Format as markdown code block
            context += f"{header}\n```{chunk['language']}\n"
            # Limit content to 1200 chars to fit in context window
            content = chunk['content'][:1200] if len(chunk['content']) > 1200 else chunk['content']
            context += f"{content}\n```\n\n"

        # Build the complete prompt for the LLM
        prompt = f"""Answer the question based on the code provided. Be concise and accurate.

{context}

Question: {query}

Answer:"""

        # Tokenize the prompt (convert text to token IDs)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",  # Return PyTorch tensors
            truncation=True,  # Truncate if too long
            max_length=2048  # Max context length
        ).to(self.model.device)  # Move to GPU if available

        # Generate response from the LLM
        with torch.no_grad():  # Disable gradient computation (we're not training)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,  # Generate up to 250 tokens
                temperature=0.3,  # Lower = more focused/deterministic
                do_sample=True,  # Use sampling (not greedy)
                top_p=0.85,  # Nucleus sampling threshold
                top_k=30,  # Consider top 30 tokens
                repetition_penalty=1.3,  # Penalize repetition
                no_repeat_ngram_size=3,  # Don't repeat 3-grams
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode tokens back to text
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer part (after "Answer:")
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            # Fallback: remove the prompt from the response
            answer = full_response[len(prompt):].strip()

        # Clean up repetitive sentences
        sentences = answer.split('. ')
        unique_sentences = []
        seen = set()
        for sent in sentences:
            sent_clean = sent.strip().lower()
            if sent_clean and sent_clean not in seen:
                seen.add(sent_clean)
                unique_sentences.append(sent.strip())

        # Reconstruct answer from unique sentences
        answer = '. '.join(unique_sentences)
        if not answer.endswith('.'):
            answer += '.'

        return answer


def main():
    """
    Main execution pipeline - orchestrates the entire RAG system
    
    This function runs the complete workflow:
    1. Clone Git repository
    2. Extract and chunk code files
    3. Create vector embeddings
    4. Load LLM
    5. Run example queries (optional)
    
    Returns:
        Tuple of (vector_store, qa_bot) for interactive use
    """

    # Configuration - change this to analyze different repositories
    REPO_URL = "https://github.com/raghavendramallela/pulumi-aws-psql-sample.git"

    # Print system information
    print("="*70)
    print("🤖 LLM CODE Q&A BOT - RAG DEMO (Updated Jan 2026)")
    print("="*70)
    print(f"Models: BGE-small-en-v1.5 + Qwen2.5-Coder-14B")
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPUs Available: {gpu_count}")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("="*70)

    # Step 1: Clone and ingest repository
    print("\n[1/5] 📥 Cloning repository...")
    ingestion = GitRepoIngestion(REPO_URL)
    ingestion.clone_repo()
    files = ingestion.get_code_files()

    if not files:
        print("❌ No code files found!")
        return None, None

    # Step 2: Parse and chunk code
    print("\n[2/5] 🔍 Parsing and chunking code...")
    chunker = CodeChunker()
    chunks = chunker.chunk_code(files)

    if not chunks:
        print("❌ No chunks created!")
        return None, None

    # Step 3: Create embeddings and vector store
    print("\n[3/5] 🧮 Creating vector store...")
    # Place embedding model on GPU 0 (RTX 4080)
    vector_store = CodeVectorStore(device="cuda:0")  # Uses BGE-small by default
    vector_store.embed_chunks(chunks)

    # Step 4: Load LLM
    print("\n[4/5] 🧠 Loading LLM...")
    # LLM will be distributed across both GPUs automatically
    qa_bot = CodeQABot(use_multi_gpu=True)  # Uses Qwen3-Coder-30B-A3B by default

    # Step 5: Interactive Q&A
    print("\n[5/5] ✅ System ready!")
    print("="*70)

    # Example queries (commented out by default)
    # Uncomment to run example questions automatically
    queries = [
        # "What does this repository do? Give me a brief overview.",
        # "How do I use the checkout action in a GitHub workflow?",
        # "Show me the main entry point and explain what it does.",
        # "What parameters can I configure for this action?"
    ]

    # Process example queries if any are defined
    for query in queries:
        print(f"\n{'='*70}")
        print(f"❓ QUERY: {query}")
        print(f"{'='*70}")

        # Retrieve relevant chunks using vector similarity search
        results = vector_store.search(query, top_k=5)

        print(f"\n📚 Retrieved {len(results)} relevant code chunks:")
        for i, chunk in enumerate(results[:3], 1):
            name_info = f" ({chunk['name']})" if chunk.get('name') != 'unknown' else ""
            print(f"  {i}. {chunk['path']}{name_info} (lines {chunk['start_line']}-{chunk['end_line']}) [score: {chunk['score']:.3f}]")

        # Generate answer using LLM
        print("\n🤖 Generating answer...\n")
        answer = qa_bot.generate_answer(query, results)
        print(answer)
        print()

    return vector_store, qa_bot

def ask_question(question: str, vector_store, qa_bot):
    """
    Ask a custom question about the codebase
    
    This function provides an improved retrieval strategy:
    - Retrieves more candidates initially (top 10)
    - Re-ranks based on question type
    - Boosts/penalizes certain file types
    
    Args:
        question: Natural language question about the code
        vector_store: CodeVectorStore instance with embeddings
        qa_bot: CodeQABot instance with loaded LLM
        
    Returns:
        Generated answer as a string
    """
    print(f"\n{'='*70}")
    print(f"❓ QUERY: {question}")
    print(f"{'='*70}")

    # Get more results initially for filtering and re-ranking
    results = vector_store.search(question, top_k=10)

    # Apply intelligent filtering based on question type
    query_lower = question.lower()
    if any(term in query_lower for term in ['explain', 'overview', 'what does', 'codebase']):
        # For overview questions, prioritize documentation and main files
        filtered = []
        for r in results:
            path_lower = r['path'].lower()
            # Boost README and main entry point files
            if 'readme' in path_lower or 'index' in path_lower or 'main' in path_lower:
                r['score'] += 0.5
            # Penalize config files (less useful for overview questions)
            if any(ext in path_lower for ext in ['.yml', '.yaml', '.json', 'package.json']):
                r['score'] -= 0.3
            filtered.append(r)

        # Re-sort by adjusted scores and take top 5
        filtered.sort(key=lambda x: x['score'], reverse=True)
        results = filtered[:5]
    else:
        # For specific questions, use top 5 as-is
        results = results[:5]

    # Display retrieved chunks
    print(f"\n📚 Found {len(results)} relevant chunks:")
    for i, chunk in enumerate(results[:3], 1):
        name_info = f" ({chunk['name']})" if chunk.get('name') != 'unknown' else ""
        print(f"  {i}. {chunk['path']}{name_info} (lines {chunk['start_line']}-{chunk['end_line']}) [score: {chunk['score']:.3f}]")

    # Generate and display answer
    print("\n🤖 ANSWER:\n")
    answer = qa_bot.generate_answer(question, results)
    print(answer)
    print()

    return answer


if __name__ == "__main__":
    """
    Entry point when script is run directly
    
    This section:
    1. Runs the main() pipeline to set up the RAG system
    2. Provides instructions for interactive use
    3. Runs an example question
    """
    # Run main pipeline to initialize everything
    vector_store, qa_bot = main()

    # Check if initialization was successful
    if vector_store and qa_bot:
        print("\n" + "="*70)
        print("✨ System Ready! You can now ask custom questions using:")
        print("   ask_question('your question here', vector_store, qa_bot)")
        print("="*70)
        # Example questions you can ask (commented out):
        # print("\nExample questions:")
        # print("  • What are all the inputs this action accepts?")
        # print("  • Show me how authentication is handled")
        # print("  • What happens when the repository doesn't exist?")
        # print("  • How does this handle submodules?")
        
# Example usage: Ask a question about the codebase
ask_question("How do i Create RDS PostgreSQL instance?",vector_store,qa_bot)
