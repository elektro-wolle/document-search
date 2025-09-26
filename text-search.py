#!/usr/bin/env python3
import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import PyPDF2
import chromadb
import fitz  # PyMuPDF - alternative PDF reader
import numpy as np
import requests
from numpy.f2py.cfuncs import includes


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a PDF with metadata"""
    filename: str
    page_number: int
    text: str
    chunk_id: str
    embedding: np.ndarray = None


class OllamaEmbedder:
    """Handles communication with Ollama for generating embeddings in batches"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text",
                 batch_size: int = 100):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.batch_size = batch_size

    def _get_embeddings_request(self, texts: List[str]) -> List[List[float]]:
        """Send a single request to Ollama for a batch of texts"""
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model,
            "input": texts if len(texts) > 1 else texts[0]
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Ollama returns {"embedding": [...]} for single, {"embeddings": [[...], ...]} for multiple
            if "embeddings" in data:  # multiple texts
                return data["embeddings"]
            else:
                raise ValueError("Unexpected response format from Ollama API")

        except Exception as e:
            print(f"Error getting embeddings: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        embedding = self._get_embeddings_request([text])[0]
        return embedding / np.linalg.norm(embedding)

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with batching"""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"Processing batch {i // self.batch_size + 1}/{(len(texts) - 1) // self.batch_size + 1}")
            batch_embeddings = self._get_embeddings_request(batch)
            embeddings.extend(batch_embeddings)
        return embeddings


class PDFProcessor:
    """Handles PDF text extraction and chunking"""

    def __init__(self, use_pymupdf: bool = True):
        self.use_pymupdf = use_pymupdf

    def extract_text_pymupdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        pages = {}

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            pages[page_num + 1] = text

        doc.close()
        return pages

    def extract_text_pypdf2(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF using PyPDF2"""
        pages = {}

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                pages[page_num + 1] = text

        return pages

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF using the selected method"""
        try:
            if self.use_pymupdf:
                return self.extract_text_pymupdf(pdf_path)
            else:
                return self.extract_text_pypdf2(pdf_path)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            # Try fallback method
            try:
                if self.use_pymupdf:
                    return self.extract_text_pypdf2(pdf_path)
                else:
                    return self.extract_text_pymupdf(pdf_path)
            except Exception as e2:
                print(f"Fallback also failed for {pdf_path}: {e2}")
                return {}

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs"""
        # Clean up text
        text = re.sub(r'\s+', ' ', text.strip())

        # Split on double newlines, periods followed by whitespace and capital letters
        # or other paragraph indicators
        paragraphs = re.split(r'(?:\n\s*\n|\. +(?=[A-Z])|(?<=\.)\s+(?=[A-Z][a-z]))', text)

        # Filter out very short paragraphs and clean up
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            # Keep paragraphs with at least 50 characters
            if len(para) >= 50:
                cleaned_paragraphs.append(para)

        return cleaned_paragraphs

    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Process a single PDF and return document chunks"""
        filename = os.path.basename(pdf_path)
        print(f"Processing {filename}...")

        pages = self.extract_text_from_pdf(pdf_path)
        chunks = []

        for page_num, page_text in pages.items():
            if not page_text.strip():
                continue

            paragraphs = self.split_into_paragraphs(page_text)

            for para_idx, paragraph in enumerate(paragraphs):
                chunk_id = f"{filename}_page{page_num}_para{para_idx + 1}"
                chunk = DocumentChunk(
                    filename=filename,
                    page_number=page_num,
                    text=paragraph,
                    chunk_id=chunk_id
                )
                chunks.append(chunk)

        return chunks


class ChromaVectorDatabase:
    """ChromaDB-based vector database"""

    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "pdf_documents"):
        self.db_path = db_path
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)

        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")

    def add_document(self, chunk: DocumentChunk, embedding: List[float]):
        """Add a document chunk to the database"""
        self.collection.add(
            embeddings=[embedding],
            documents=[chunk.text],
            metadatas=[{
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id
            }],
            ids=[chunk.chunk_id]
        )

    def add_documents_batch(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add multiple document chunks in batch"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [{
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id
        } for chunk in chunks]

        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))

            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            print(f"Added batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

    # --- Utility ---
    @staticmethod
    def cosine_similarity(a, b):
        # noinspection PyTypeChecker
        norm_a: float = np.linalg.norm(a)
        # noinspection PyTypeChecker
        norm_b: float = np.linalg.norm(b)

        return np.dot(a, b) / (norm_a * norm_b)

    def search_with_filters(
            self,
            include_embeddings: list[list[float]],
            exclude_embeddings: list[list[float]] = None,
            top_k: int = 10,
            penalty_lambda: float = 1.0,
            exclude_threshold: float = None,
            fetch_k: int = 50,
    ):
        """
        Query ChromaDB with include and exclude embeddings.

        Args:
`           include_embeddings: list of concepts to include
            exclude_embeddings: list of concepts to exclude
            top_k: number of results to return
            penalty_lambda: weight for penalizing exclude similarity
            exclude_threshold: if set, drops docs with similarity above this
            fetch_k: how many candidates to fetch from ChromaDB before filtering

        Returns:
            List of (document, score) tuples
        """

        # Create include embedding
        include_vector = np.mean(include_embeddings, axis=0)
        include_vector = include_vector / np.linalg.norm(include_vector)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[include_vector.tolist()],
            n_results=fetch_k,
            include=['documents', 'embeddings', 'metadatas'],
        )

        # Post-process results
        reranked = []
        for doc, emb, meta in zip(
                results["documents"][0],
                results["embeddings"][0],
                results["metadatas"][0],
            ):
            sim_incl = ChromaVectorDatabase.cosine_similarity(include_vector, emb)

            if exclude_embeddings:
                sim_excl = [ChromaVectorDatabase.cosine_similarity(emb, e) for e in exclude_embeddings]
                max_excl = max(sim_excl)

                # Apply hard cutoff if threshold is set
                if exclude_threshold is not None and max_excl >= exclude_threshold:
                    continue

                # Penalize based on similarity to excludes
                final_score = sim_incl - penalty_lambda * max_excl
            else:
                final_score = sim_incl

            result_doc = {
                'document': doc,
                'metadata': meta,
                'embedding': emb,
                'score': final_score,
            }

            reranked.append(result_doc)

        # Sort and return top_k
        reranked = sorted(reranked, key=lambda x: x['score'] ,reverse=True)
        return reranked[:top_k]

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "db_path": self.db_path
        }


class PDFVectorSearchSystem:
    """Main system that combines all components"""

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "nomic-embed-text",
                 db_path: str = "./chroma_db",
                 collection_name: str = "pdf_documents"):
        self.embedder = OllamaEmbedder(ollama_url, embedding_model)
        self.pdf_processor = PDFProcessor()
        self.vector_db = ChromaVectorDatabase(db_path, collection_name)

    def ingest_pdfs(self, pdf_directory: str):
        """Process all PDFs in a directory and add them to the vector database"""
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))

        if not pdf_files:
            print("No PDF files found in the specified directory.")
            return
        pdf_files.sort()
        print(f"Found {len(pdf_files)} PDF files to process.")

        for pdf_path in pdf_files:
            try:
                # Extract text and create chunks
                chunks = self.pdf_processor.process_pdf(str(pdf_path))

                if not chunks:
                    print(f"No text chunks extracted from {pdf_path.name}")
                    continue

                print(f"Extracted {len(chunks)} chunks from {pdf_path.name}")

                # Generate embeddings
                texts = [chunk.text for chunk in chunks]
                embeddings = self.embedder.get_embeddings_batch(texts)

                # Add to ChromaDB in batch
                self.vector_db.add_documents_batch(chunks, embeddings)

                print(f"Successfully processed and indexed {pdf_path.name}")

            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")

    def search(self, include: List[str], exclude: List[str] = None,
               similarity_threshold: float = 0.3, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents using include/exclude filters
        
        Args:
            include: List of terms that documents MUST match (AND condition)
            exclude: List of terms that documents must NOT match (OR condition)
            similarity_threshold: Minimum similarity score to consider a match
            top_k: Maximum number of results to return
        """
        if not include:
            raise ValueError("At least one include term is required")

        if exclude is None:
            exclude = []

        print(f"Searching with include terms: {include}")
        if exclude:
            print(f"Excluding terms: {exclude}")

        # Get embeddings for include terms (AND condition)
        include_embeddings = []
        for term in include:
            embedding = self.embedder.get_embedding(term)
            include_embeddings.append(embedding)

        # Get embeddings for exclude terms (OR condition)
        exclude_embeddings = []
        for term in exclude:
            embedding = self.embedder.get_embedding(term)
            exclude_embeddings.append(embedding)

        # Search in vector database
        results = self.vector_db.search_with_filters(
            include_embeddings, exclude_embeddings,
            top_k=10,
            fetch_k=100
        )

        # Format results
        formatted_results = []
        for i, result_doc in enumerate(results):
            # Calculate individual similarities for display
            doc_embedding = np.array(result_doc['embedding'])

            include_sims = []
            for include_emb in include_embeddings:
                include_emb_np = np.array(include_emb)
                dot_product = np.dot(doc_embedding, include_emb_np)
                norm_product = np.linalg.norm(doc_embedding) * np.linalg.norm(include_emb_np)
                sim = dot_product / norm_product if norm_product > 0 else 0
                include_sims.append(sim)

            exclude_sims = []
            for exclude_emb in exclude_embeddings:
                exclude_emb_np = np.array(exclude_emb)
                dot_product = np.dot(doc_embedding, exclude_emb_np)
                norm_product = np.linalg.norm(doc_embedding) * np.linalg.norm(exclude_emb_np)
                sim = dot_product / norm_product if norm_product > 0 else 0
                exclude_sims.append(sim)

            text = result_doc['document']
            formatted_results.append({
                'rank': i + 1,
                'filename': result_doc['metadata']['filename'],
                'page_number': result_doc['metadata']['page_number'],
                'overall_score': float(result_doc['score']),
                'include_similarities': {term: float(sim) for term, sim in zip(include, include_sims)},
                'exclude_similarities': {term: float(sim) for term, sim in
                                         zip(exclude, exclude_sims)} if exclude else {},
                'text': text[:500] + "..." if len(text) > 500 else text,
                'full_text': text
            })

        return formatted_results

    def print_search_results(self, results: List[Dict[str, Any]]):
        """Pretty print search results with include/exclude details"""
        if not results:
            print("No results found.")
            return

        print(f"\nFound {len(results)} results:\n")
        print("=" * 80)

        for result in results:
            print(f"Rank: {result['rank']}")
            print(f"File: {result['filename']}")
            print(f"Page: {result['page_number']}")
            print(f"Overall Score: {result['overall_score']:.4f}")

            # Show include similarities
            if result['include_similarities']:
                print("Include term similarities:")
                for term, sim in result['include_similarities'].items():
                    print(f"  '{term}': {sim:.4f}")

            # Show exclude similarities (if any)
            if result['exclude_similarities']:
                print("Exclude term similarities:")
                for term, sim in result['exclude_similarities'].items():
                    print(f"  '{term}': {sim:.4f}")

            print(f"Text: {result['text']}")
            print("-" * 80)

    def get_database_info(self):
        """Print database information"""
        info = self.vector_db.get_collection_info()
        print(f"Database: {info['db_path']}")
        print(f"Collection: {info['collection_name']}")
        print(f"Document count: {info['document_count']}")


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="PDF Vector Search System with Ollama embeddings and ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest PDFs from a directory
  python script.py ingest --directory /path/to/pdfs

  # Search with include terms only
  python script.py search --include "machine learning" "algorithms"

  # Search with include and exclude terms
  python script.py search --include "AI" "python" --exclude "java" "neural networks"

  # Interactive mode
  python script.py interactive

  # Check database info
  python script.py info
        """
    )

    # Global options
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--embedding-model", default="nomic-embed-text",
                        help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--db-path", default="./chroma_db",
                        help="ChromaDB database path (default: ./chroma_db)")
    parser.add_argument("--collection", default="pdf_documents",
                        help="ChromaDB collection name (default: pdf_documents)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDFs from directory")
    ingest_parser.add_argument("--directory", required=True,
                               help="Directory containing PDF files")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("--include", nargs="+", required=True,
                               help="Include terms (AND condition)")
    search_parser.add_argument("--exclude", nargs="*", default=[],
                               help="Exclude terms (OR condition)")
    search_parser.add_argument("--threshold", type=float, default=0.3,
                               help="Similarity threshold (default: 0.3)")
    search_parser.add_argument("--top-k", type=int, default=10,
                               help="Maximum number of results (default: 10)")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive mode")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show database information")

    return parser


def interactive_mode(search_system: PDFVectorSearchSystem):
    """Run interactive command-line interface"""
    print("PDF Vector Search System - Interactive Mode")
    print("Commands:")
    print("  ingest <directory> - Process PDFs from directory")
    print("  search --include term1,term2 --exclude term3,term4 - Search with filters")
    print("  search term1,term2 - Search with only include terms")
    print("  info - Show database information")
    print("  quit - Exit")

    while True:
        try:
            command = input("\n> ").strip()

            if command.lower() in ['quit', 'exit']:
                break
            elif command.startswith('ingest '):
                directory = command[7:].strip()
                if directory:
                    search_system.ingest_pdfs(directory)
                else:
                    print("Please specify a directory")
            elif command.startswith('search '):
                # Parse search command
                search_part = command[7:].strip()

                include_terms = []
                exclude_terms = []

                if '--include' in search_part or '--exclude' in search_part:
                    # Parse with flags
                    parts = search_part.split('--')
                    for part in parts:
                        part = part.strip()
                        if part.startswith('include '):
                            terms_str = part[8:].strip()
                            include_terms = [t.strip() for t in terms_str.split(',') if t.strip()]
                        elif part.startswith('exclude '):
                            terms_str = part[8:].strip()
                            exclude_terms = [t.strip() for t in terms_str.split(',') if t.strip()]
                else:
                    # Simple format - just include terms
                    include_terms = [t.strip() for t in search_part.split(',') if t.strip()]

                if include_terms:
                    # try:
                    results = search_system.search(include_terms, exclude_terms, top_k=10)
                    search_system.print_search_results(results)
                # except Exception as e:
                #     print(f"Search error: {e}")
                else:
                    print("Please provide at least one include term.")
            elif command.lower() == 'info':
                search_system.get_database_info()
            elif command.strip():
                print("Unknown command.")
                print("Examples:")
                print("  search machine learning,algorithms")
                print("  search --include machine learning,AI --exclude neural networks,deep learning")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        # except Exception as e:
        #     print(f"Error: {e}")


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Initialize the system
    search_system = PDFVectorSearchSystem(
        ollama_url=args.ollama_url,
        embedding_model=args.embedding_model,
        db_path=args.db_path,
        collection_name=args.collection
    )

    if args.command == "ingest":
        search_system.ingest_pdfs(args.directory)
    elif args.command == "search":
        # try:
        results = search_system.search(
            include=args.include,
            exclude=args.exclude,
            similarity_threshold=args.threshold,
            top_k=args.top_k
        )
        search_system.print_search_results(results)
    # except Exception as e:
    #     print(f"Search error: {e}")
    elif args.command == "interactive":
        interactive_mode(search_system)
    elif args.command == "info":
        search_system.get_database_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
