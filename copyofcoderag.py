import sys
if sys.version_info < (3, 10):
    print("⚠️  Warning: Python 3.10+ recommended. Current:", sys.version)

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import subprocess

import git
from tree_sitter_languages import get_parser, get_language
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print(f"PyTorch version: {torch.__version__}")
if torch.__version__ < "2.1.0":
    print("⚠️  Warning: PyTorch 2.3+ recommended for transformers v5.0")

class GitRepoIngestion:
    """Clone and extract code files from Git repository"""

    def __init__(self, repo_url: str, local_path: str = "./repo"):
        self.repo_url = repo_url
        self.local_path = local_path
        self.code_extensions = {
            '.js', '.ts', '.tsx', '.jsx', '.py', '.java', '.go', '.rb',
            '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.sh', '.bash',
            '.yml', '.yaml', '.json', '.md', '.rs', '.swift', '.kt'
        }

    def clone_repo(self):
        """Clone the repository"""
        if os.path.exists(self.local_path):
            print(f"Repository already exists at {self.local_path}")
            return git.Repo(self.local_path)

        print(f"Cloning {self.repo_url}...")
        repo = git.Repo.clone_from(self.repo_url, self.local_path)
        print(f"✓ Cloned successfully")
        return repo

    def get_code_files(self) -> List[Dict[str, str]]:
        """Extract all code files with content"""
        files = []
        exclude_dirs = {
            '.git', 'node_modules', '__pycache__', 'dist', 'build',
            '.venv', 'venv', '.pytest_cache', '.mypy_cache', 'target',
            'bin', 'obj', '.gradle'
        }

        for root, dirs, filenames in os.walk(self.local_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for filename in filenames:
                file_path = Path(root) / filename
                if file_path.suffix in self.code_extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        if len(content) > 500_000:
                            continue

                        relative_path = file_path.relative_to(self.local_path)
                        files.append({
                            'path': str(relative_path),
                            'content': content,
                            'language': file_path.suffix[1:]
                        })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        print(f"✓ Found {len(files)} code files")
        return files

class CodeChunker:
    """Parse and chunk code into meaningful segments"""

    def __init__(self):
        self.parser_cache = {}

    def get_parser(self, language: str):
        """Get tree-sitter parser for language"""
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
            'h': 'cpp',
            'hpp': 'cpp',
            'cs': 'c_sharp',
            'sh': 'bash',
            'bash': 'bash',
            'rs': 'rust',
            'php': 'php'
        }

        lang = lang_map.get(language, language)
        if lang not in self.parser_cache:
            try:
                self.parser_cache[lang] = get_parser(lang)
            except Exception as e:
                print(f"Parser not available for {lang}: {e}")
                return None
        return self.parser_cache.get(lang)

    def chunk_code(self, files: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Create chunks from code files"""
        chunks = []

        for file in files:
            parser = self.get_parser(file['language'])

            if parser and file['language'] in ['py', 'js', 'ts', 'jsx', 'tsx', 'java', 'go', 'rs']:
                file_chunks = self._parse_with_tree_sitter(file, parser)
                if file_chunks:
                    chunks.extend(file_chunks)
                else:
                    chunks.extend(self._simple_chunk(file))
            else:
                chunks.extend(self._simple_chunk(file))

        print(f"✓ Created {len(chunks)} code chunks")
        return chunks

    def _parse_with_tree_sitter(self, file: Dict, parser) -> List[Dict]:
        """Parse file with tree-sitter"""
        chunks = []
        try:
            tree = parser.parse(bytes(file['content'], 'utf8'))

            self._extract_nodes(tree.root_node, file, chunks)

            if not chunks:
                return []

        except Exception as e:
            print(f"Parser error for {file['path']}: {e}")
            return []

        return chunks

    def _extract_nodes(self, node, file: Dict, chunks: List):
        """Recursively extract function and class nodes"""
        interesting_types = {
            'function_definition', 'function_declaration', 'function_item',
            'class_definition', 'class_declaration', 'class_item',
            'method_definition', 'method_declaration',
            'interface_declaration', 'struct_item',
            'impl_item',
        }

        if node.type in interesting_types:
            code = file['content'][node.start_byte:node.end_byte]
            name = self._extract_name(node, file['content'])

            chunks.append({
                'content': code,
                'path': file['path'],
                'language': file['language'],
                'type': node.type,
                'name': name,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1
            })

        for child in node.children:
            self._extract_nodes(child, file, chunks)

    def _extract_name(self, node, content: str) -> str:
        """Extract name from node"""
        try:
            for child in node.children:
                if 'identifier' in child.type or child.type == 'name':
                    return content[child.start_byte:child.end_byte]
        except:
            pass
        return "unknown"

    def _simple_chunk(self, file: Dict) -> List[Dict]:
        """Simple chunking by size with overlap"""
        content = file['content']
        lines = content.split('\n')

        if 'readme' in file['path'].lower():
            chunk_size = 200
            overlap = 20
        else:
            chunk_size = 80
            overlap = 15

        chunks = []

        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
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
    """Embed code chunks and enable similarity search"""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize with better embedding model"""
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(
            model_name,
            trust_remote_code=True
        )
        self.chunks = []
        self.index = None
        print(f"✓ Embedding model loaded (dim: {self.embedding_model.get_sentence_embedding_dimension()})")

    def embed_chunks(self, chunks: List[Dict[str, Any]]):
        """Generate embeddings for all chunks"""
        self.chunks = chunks

        print(f"Generating embeddings for {len(chunks)} chunks...")

        texts = []
        for c in chunks:
            metadata = f"File: {c['path']} | Language: {c['language']}"
            if c.get('name') and c['name'] != 'unknown':
                metadata += f" | {c['type']}: {c['name']}"

            text = f"{metadata}\n\n{c['content'][:2000]}"
            texts.append(text)

        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))

        print(f"✓ Vector store created with {self.index.ntotal} embeddings")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant code chunks"""
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(score)
                results.append(chunk)

        return results

class CodeQABot:
    """LLM-based Q&A bot for code"""

    def __init__(self, model_name: str = "codellama/CodeLlama-7b-hf"):
        """Initialize with code LLM"""
        print(f"Loading LLM: {model_name}...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✓ LLM loaded (4-bit quantized, {model_name})")
        print(f"  Memory footprint: ~{torch.cuda.memory_allocated() / 1024**3:.1f}GB")

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using retrieved context"""

        context = "# CODE CONTEXT:\n\n"
        for chunk in context_chunks[:3]:
            header = f"## File: {chunk['path']} (Lines {chunk['start_line']}-{chunk['end_line']})"
            if chunk.get('name') and chunk['name'] != 'unknown':
                header += f" - {chunk['name']}"

            context += f"{header}\n```{chunk['language']}\n"
            content = chunk['content'][:1200] if len(chunk['content']) > 1200 else chunk['content']
            context += f"{content}\n```\n\n"

        prompt = f"""Answer the question based on the code provided. Be concise and accurate.

{context}

Question: {query}

Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.3,
                do_sample=True,
                top_p=0.85,
                top_k=30,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            answer = full_response[len(prompt):].strip()

        sentences = answer.split('. ')
        unique_sentences = []
        seen = set()
        for sent in sentences:
            sent_clean = sent.strip().lower()
            if sent_clean and sent_clean not in seen:
                seen.add(sent_clean)
                unique_sentences.append(sent.strip())

        answer = '. '.join(unique_sentences)
        if not answer.endswith('.'):
            answer += '.'

        return answer

def main():
    """Main execution pipeline"""

    REPO_URL = "https://github.com/raghavendramallela/pulumi-aws-psql-sample.git"

    print("="*70)
    print("🤖 LLM CODE Q&A BOT - RAG DEMO")
    print("="*70)
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)

    print("\n[1/5] 📥 Cloning repository...")
    ingestion = GitRepoIngestion(REPO_URL)
    ingestion.clone_repo()
    files = ingestion.get_code_files()

    if not files:
        print("❌ No code files found!")
        return None, None

    print("\n[2/5] 🔍 Parsing and chunking code...")
    chunker = CodeChunker()
    chunks = chunker.chunk_code(files)

    if not chunks:
        print("❌ No chunks created!")
        return None, None

    print("\n[3/5] 🧮 Creating vector store...")
    vector_store = CodeVectorStore()
    vector_store.embed_chunks(chunks)

    print("\n[4/5] 🧠 Loading LLM...")
    qa_bot = CodeQABot()

    print("\n[5/5] ✅ System ready!")
    print("="*70)

    return vector_store, qa_bot

def ask_question(question: str, vector_store, qa_bot):
    """Ask a custom question with improved retrieval"""
    print(f"\n{'='*70}")
    print(f"❓ QUERY: {question}")
    print(f"{'='*70}")

    results = vector_store.search(question, top_k=10)

    query_lower = question.lower()
    if any(term in query_lower for term in ['explain', 'overview', 'what does', 'codebase']):
        filtered = []
        for r in results:
            path_lower = r['path'].lower()
            if 'readme' in path_lower or 'index' in path_lower or 'main' in path_lower:
                r['score'] += 0.5
            if any(ext in path_lower for ext in ['.yml', '.yaml', '.json', 'package.json']):
                r['score'] -= 0.3
            filtered.append(r)

        filtered.sort(key=lambda x: x['score'], reverse=True)
        results = filtered[:5]
    else:
        results = results[:5]

    print(f"\n📚 Found {len(results)} relevant chunks:")
    for i, chunk in enumerate(results[:3], 1):
        name_info = f" ({chunk['name']})" if chunk.get('name') != 'unknown' else ""
        print(f"  {i}. {chunk['path']}{name_info} (lines {chunk['start_line']}-{chunk['end_line']}) [score: {chunk['score']:.3f}]")

    print("\n🤖 ANSWER:\n")
    answer = qa_bot.generate_answer(question, results)
    print(answer)
    print()

    return answer

if __name__ == "__main__":
    vector_store, qa_bot = main()

    if vector_store and qa_bot:
        print("\n" + "="*70)
        print("✨ System Ready! You can now ask custom questions using:")
        print("   ask_question('your question here', vector_store, qa_bot)")
        print("="*70)

        # Example usage
        ask_question("How do i Create RDS PostgreSQL instance?", vector_store, qa_bot)
