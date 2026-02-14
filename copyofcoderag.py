import sys

# Warn user if Python version is older than recommended.
# Some libraries used later (transformers, torch) benefit from newer Python versions.
if sys.version_info < (3, 10):
    print("⚠️  Warning: Python 3.10+ recommended. Current:", sys.version)

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import subprocess

# GitPython for cloning repositories
import git

# Tree-sitter allows AST-level parsing of code across many languages
from tree_sitter_languages import get_parser, get_language

import numpy as np
import faiss  # High-performance vector similarity search library

# SentenceTransformer generates semantic embeddings for code chunks
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Show PyTorch version and warn if outdated for quantized transformer models
print(f"PyTorch version: {torch.__version__}")
if torch.__version__ < "2.1.0":
    print("⚠️  Warning: PyTorch 2.3+ recommended for transformers v5.0")


class GitRepoIngestion:
    """Clone and extract code files from Git repository"""

    def __init__(self, repo_url: str, local_path: str = "./repo"):
        # Repository to download
        self.repo_url = repo_url

        # Local directory where repo will be stored
        self.local_path = local_path

        # Allowed file extensions to treat as "code"
        self.code_extensions = {
            '.js', '.ts', '.tsx', '.jsx', '.py', '.java', '.go', '.rb',
            '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.sh', '.bash',
            '.yml', '.yaml', '.json', '.md', '.rs', '.swift', '.kt'
        }

    def clone_repo(self):
        """Clone the repository if not already present"""

        # Avoid re-cloning if directory exists
        if os.path.exists(self.local_path):
            print(f"Repository already exists at {self.local_path}")
            return git.Repo(self.local_path)

        print(f"Cloning {self.repo_url}...")
        repo = git.Repo.clone_from(self.repo_url, self.local_path)
        print(f"✓ Cloned successfully")
        return repo

    def get_code_files(self) -> List[Dict[str, str]]:
        """Walk repository and load code file contents"""

        files = []

        # Directories commonly containing build artifacts or dependencies
        exclude_dirs = {
            '.git', 'node_modules', '__pycache__', 'dist', 'build',
            '.venv', 'venv', '.pytest_cache', '.mypy_cache', 'target',
            'bin', 'obj', '.gradle'
        }

        # Recursively walk repo
        for root, dirs, filenames in os.walk(self.local_path):

            # Remove excluded directories in-place to prevent traversal
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for filename in filenames:
                file_path = Path(root) / filename

                # Only process recognized code file types
                if file_path.suffix in self.code_extensions:
                    try:
                        # Ignore encoding errors to avoid crashes on odd files
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        # Skip extremely large files to avoid memory issues
                        if len(content) > 500_000:
                            continue

                        relative_path = file_path.relative_to(self.local_path)

                        files.append({
                            'path': str(relative_path),
                            'content': content,
                            'language': file_path.suffix[1:]  # remove dot
                        })

                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        print(f"✓ Found {len(files)} code files")
        return files


class CodeChunker:
    """Parse and chunk code into meaningful segments"""

    def __init__(self):
        # Cache parsers to avoid expensive reloads
        self.parser_cache = {}

    def get_parser(self, language: str):
        """Return tree-sitter parser for given language"""

        # Map common extensions to tree-sitter language names
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

        # Load parser only once
        if lang not in self.parser_cache:
            try:
                self.parser_cache[lang] = get_parser(lang)
            except Exception as e:
                print(f"Parser not available for {lang}: {e}")
                return None

        return self.parser_cache.get(lang)

    def chunk_code(self, files: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Split files into retrievable chunks"""

        chunks = []

        for file in files:
            parser = self.get_parser(file['language'])

            # Prefer AST-based chunking when supported
            if parser and file['language'] in ['py', 'js', 'ts', 'jsx', 'tsx', 'java', 'go', 'rs']:
                file_chunks = self._parse_with_tree_sitter(file, parser)

                # Fall back to simple chunking if parsing fails
                if file_chunks:
                    chunks.extend(file_chunks)
                else:
                    chunks.extend(self._simple_chunk(file))
            else:
                chunks.extend(self._simple_chunk(file))

        print(f"✓ Created {len(chunks)} code chunks")
        return chunks

    def _parse_with_tree_sitter(self, file: Dict, parser) -> List[Dict]:
        """Use AST to extract functions/classes instead of naive slicing"""

        chunks = []
        try:
            tree = parser.parse(bytes(file['content'], 'utf8'))

            # Recursively walk nodes
            self._extract_nodes(tree.root_node, file, chunks)

            if not chunks:
                return []

        except Exception as e:
            print(f"Parser error for {file['path']}: {e}")
            return []

        return chunks

    def _extract_nodes(self, node, file: Dict, chunks: List):
        """Recursively collect interesting structural nodes"""

        # Node types that usually represent meaningful retrieval units
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

        # Continue traversing child nodes
        for child in node.children:
            self._extract_nodes(child, file, chunks)

    def _extract_name(self, node, content: str) -> str:
        """Attempt to pull identifier (function/class name)"""

        try:
            for child in node.children:
                if 'identifier' in child.type or child.type == 'name':
                    return content[child.start_byte:child.end_byte]
        except:
            pass

        return "unknown"

    def _simple_chunk(self, file: Dict) -> List[Dict]:
        """
        Fallback chunking strategy:
        Split by line count with overlap to preserve context.
        """

        content = file['content']
        lines = content.split('\n')

        # Larger chunks for READMEs (more narrative)
        if 'readme' in file['path'].lower():
            chunk_size = 200
            overlap = 20
        else:
            chunk_size = 80
            overlap = 15

        chunks = []

        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]

            # Skip empty chunks
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