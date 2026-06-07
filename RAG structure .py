# =============================================================
#  SIMPLE RAG — from scratch, zero external dependencies
#  (uses numpy only for vector math)
#
#  STRUCTURE:
#  1. DOCUMENTS  — your knowledge base (raw text)
#  2. CHUNKING   — split docs into smaller pieces
#  3. EMBEDDING  — convert chunks to vectors (numbers)
#  4. VECTOR STORE — store & search vectors
#  5. RETRIEVER  — find relevant chunks for a query
#  6. GENERATOR  — combine query + chunks → answer (LLM call)
#  7. RAG PIPELINE — wire all pieces together
# =============================================================

import numpy as np
import re
from typing import List, Tuple


# ─────────────────────────────────────────────
# 1. DOCUMENTS  — your knowledge base
# ─────────────────────────────────────────────
# In a real system these would be PDFs, websites, databases.
# Here we use plain strings to keep it simple.

DOCUMENTS = [
    """
    Python is a high-level programming language created by Guido van Rossum in 1991.
    It emphasizes code readability and uses indentation to define code blocks.
    Python supports multiple programming paradigms including procedural, object-oriented,
    and functional programming. It is widely used in web development, data science,
    artificial intelligence, and automation.
    """,
    """
    Machine learning is a subset of artificial intelligence that enables systems to learn
    from data without being explicitly programmed. Common types include supervised learning,
    unsupervised learning, and reinforcement learning. Popular ML frameworks include
    TensorFlow, PyTorch, and scikit-learn.
    """,
    """
    RAG stands for Retrieval-Augmented Generation. It is a technique that combines
    information retrieval with text generation. First, relevant documents are retrieved
    from a knowledge base using vector similarity search. Then, those documents are
    passed to a language model to generate an accurate, grounded answer.
    RAG helps reduce hallucinations in AI systems.
    """,
    """
    Large Language Models (LLMs) are deep learning models trained on vast amounts of
    text data. Examples include GPT-4, Claude, and Gemini. They can perform tasks like
    text generation, summarization, translation, and question answering. LLMs use the
    transformer architecture with attention mechanisms.
    """,
    """
    Vector databases store data as high-dimensional vectors (embeddings). They allow
    fast similarity search using algorithms like cosine similarity or dot product.
    Popular vector databases include Pinecone, Weaviate, Chroma, and Qdrant.
    They are the core storage layer in RAG systems.
    """,
]


# ─────────────────────────────────────────────
# 2. CHUNKING — split text into smaller pieces
# ─────────────────────────────────────────────
# Why chunk? LLMs have limited context windows.
# Smaller chunks also improve retrieval precision.

def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    """
    Split text into chunks of roughly `chunk_size` words.
    In production: use sentence boundaries, overlap, or sliding windows.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def build_chunks(documents: List[str]) -> List[str]:
    """Chunk all documents and return a flat list of chunks."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
    return all_chunks


# ─────────────────────────────────────────────
# 3. EMBEDDING — convert text to vectors
# ─────────────────────────────────────────────
# In production: use OpenAI text-embedding-ada, sentence-transformers, etc.
# Here we use a simple bag-of-words TF embedding so you can run this
# without any API keys. The structure is identical to real embedding.

def build_vocab(chunks: List[str]) -> List[str]:
    """Build a vocabulary from all chunks."""
    words = set()
    for chunk in chunks:
        for w in re.findall(r'\w+', chunk.lower()):
            words.add(w)
    return sorted(list(words))


def embed_text(text: str, vocab: List[str]) -> np.ndarray:
    """
    Convert text → vector using simple word frequency (bag-of-words).
    Real embeddings (e.g. from sentence-transformers) capture semantic meaning.
    This demo version still works for exact/partial keyword matching.
    """
    word_counts = {}
    for w in re.findall(r'\w+', text.lower()):
        word_counts[w] = word_counts.get(w, 0) + 1

    vector = np.zeros(len(vocab), dtype=float)
    for i, word in enumerate(vocab):
        vector[i] = word_counts.get(word, 0)

    # Normalize to unit length (required for cosine similarity)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector


# ─────────────────────────────────────────────
# 4. VECTOR STORE — store & search embeddings
# ─────────────────────────────────────────────
# In production: Pinecone, Chroma, Weaviate, etc.
# Here: a simple in-memory list of (chunk, vector) pairs.

class SimpleVectorStore:
    """
    Minimal in-memory vector store.
    Stores chunks with their embeddings and supports similarity search.
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.vectors: List[np.ndarray] = []

    def add(self, chunk: str, vector: np.ndarray):
        """Add a chunk and its embedding to the store."""
        self.chunks.append(chunk)
        self.vectors.append(vector)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find the top_k most similar chunks using cosine similarity.
        Cosine similarity = dot product of two unit vectors.
        Score ranges from 0 (unrelated) to 1 (identical).
        """
        scores = []
        for i, vector in enumerate(self.vectors):
            score = float(np.dot(query_vector, vector))  # cosine similarity
            scores.append((self.chunks[i], score))

        # Sort by score descending, return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def __len__(self):
        return len(self.chunks)


# ─────────────────────────────────────────────
# 5. RETRIEVER — find relevant chunks for a query
# ─────────────────────────────────────────────

class Retriever:
    """
    Takes a query, embeds it, searches the vector store,
    and returns the most relevant chunks.
    """

    def __init__(self, vector_store: SimpleVectorStore, vocab: List[str]):
        self.vector_store = vector_store
        self.vocab = vocab

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Embed the query and find the closest chunks."""
        query_vector = embed_text(query, self.vocab)
        results = self.vector_store.search(query_vector, top_k=top_k)
        return results


# ─────────────────────────────────────────────
# 6. GENERATOR — combine query + chunks → answer
# ─────────────────────────────────────────────
# In production: call Claude, GPT-4, or any LLM API here.
# Here: we simulate the LLM by extracting relevant sentences
# from the retrieved chunks (so you don't need an API key).

class Generator:
    """
    Combines the retrieved context with the query to produce an answer.
    In production: this sends a prompt to Claude/GPT and returns its response.
    """

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """
        Build a prompt from the query + retrieved chunks, then call the LLM.
        This shows the EXACT prompt structure used in real RAG systems.
        """

        # --- This is the actual RAG prompt structure ---
        context = "\n\n".join(
            [f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)]
        )

        prompt = f"""
You are a helpful assistant. Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't know."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        # In production you would do:
        # response = anthropic.messages.create(model="claude-sonnet-4-20250514",
        #                                      messages=[{"role":"user","content":prompt}])
        # return response.content[0].text

        # Demo: simulate LLM by picking the most relevant sentence from context
        return self._simulate_llm(query, context_chunks, prompt)

    def _simulate_llm(self, query: str, chunks: List[str], prompt: str) -> str:
        """
        Simulated LLM response (keyword matching).
        Replace this entire method with a real API call in production.
        """
        query_words = set(re.findall(r'\w+', query.lower()))

        best_sentence = ""
        best_score = 0

        for chunk in chunks:
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                sent_words = set(re.findall(r'\w+', sentence.lower()))
                score = len(query_words & sent_words)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence

        if best_sentence:
            return f"{best_sentence} [Retrieved from knowledge base — in production this would be a full LLM-generated answer]"
        return "I don't know based on the available context."


# ─────────────────────────────────────────────
# 7. RAG PIPELINE — wire everything together
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    The full RAG system.

    INDEXING phase  (runs once at startup):
      Documents → Chunks → Embeddings → Vector Store

    QUERYING phase  (runs on every user question):
      Query → Embed → Retrieve chunks → Generate answer
    """

    def __init__(self, documents: List[str]):
        print("=" * 55)
        print("  RAG PIPELINE — INDEXING PHASE")
        print("=" * 55)

        # Step 1: Chunk all documents
        print("\n[1] Chunking documents...")
        self.chunks = build_chunks(documents)
        print(f"    Created {len(self.chunks)} chunks from {len(documents)} documents")

        # Step 2: Build vocabulary (needed for our demo embedder)
        print("\n[2] Building vocabulary...")
        self.vocab = build_vocab(self.chunks)
        print(f"    Vocabulary size: {len(self.vocab)} unique words")

        # Step 3: Embed all chunks and store them
        print("\n[3] Embedding chunks → storing in vector store...")
        self.vector_store = SimpleVectorStore()
        for chunk in self.chunks:
            vector = embed_text(chunk, self.vocab)
            self.vector_store.add(chunk, vector)
        print(f"    Stored {len(self.vector_store)} vectors")

        # Step 4: Set up retriever and generator
        self.retriever = Retriever(self.vector_store, self.vocab)
        self.generator = Generator()

        print("\n  Indexing complete. Ready for queries.")
        print("=" * 55)

    def query(self, user_question: str, top_k: int = 3) -> str:
        """
        Full RAG query flow:
        question → embed → retrieve → generate → answer
        """
        print(f"\n{'─'*55}")
        print(f"  QUERY: {user_question}")
        print(f"{'─'*55}")

        # Step 1: Retrieve relevant chunks
        print(f"\n[Retrieve] Searching vector store (top {top_k})...")
        results = self.retriever.retrieve(user_question, top_k=top_k)

        print("\n  Retrieved chunks:")
        context_chunks = []
        for i, (chunk, score) in enumerate(results):
            preview = chunk[:70] + "..." if len(chunk) > 70 else chunk
            print(f"  {i+1}. score={score:.3f} | {preview}")
            context_chunks.append(chunk)

        # Step 2: Generate answer using retrieved context
        print("\n[Generate] Sending context + query to LLM...")
        answer = self.generator.generate(user_question, context_chunks)

        print(f"\n  ANSWER: {answer}")
        return answer


# ─────────────────────────────────────────────
# MAIN — run the pipeline with example queries
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Build the RAG pipeline (indexing phase)
    rag = RAGPipeline(DOCUMENTS)

    # Ask questions (querying phase)
    questions = [
        "What is RAG and how does it work?",
        "What programming language did Guido van Rossum create?",
        "What are vector databases used for?",
        "What is machine learning?",
    ]

    for question in questions:
        rag.query(question)

    print(f"\n{'='*55}")
    print("  To use a real LLM (Claude), replace _simulate_llm()")
    print("  with an Anthropic API call. Everything else stays the same.")
    print("=" * 55)