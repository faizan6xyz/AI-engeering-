
LLM_TERMS = {

    "LLM (Large Language Model)": {
        "explanation": (
            "A neural network trained on massive amounts of text data to understand "
            "and generate human language. It predicts the next token given previous tokens."
        ),
        "use_case": "Chatbots, code generation, summarization, translation, Q&A.",
        "examples": "GPT-4, Claude, Gemini, Llama 3, Qwen2.5, Mistral",
    },

    "Token": {
        "explanation": (
            "The basic unit of text that an LLM processes. A token is roughly 3-4 "
            "characters or 0.75 words. Text is split into tokens before being fed to the model."
        ),
        "use_case": "Pricing API calls, measuring context window usage, understanding model limits.",
        "examples": "'Hello world' = 2 tokens. 'unbelievable' = 3 tokens.",
    },

    "Context Window": {
        "explanation": (
            "The maximum number of tokens an LLM can process in one call — including "
            "both the input (prompt) and the output (response). Beyond this limit, the "
            "model cannot see earlier content."
        ),
        "use_case": "Determines how much text you can send at once — affects RAG chunk size, conversation memory.",
        "examples": "GPT-4: 128K tokens. Claude: 200K tokens. Gemini 1.5: 1M tokens.",
    },

    "Prompt": {
        "explanation": (
            "The input text you send to an LLM. It tells the model what to do. "
            "Prompts can include instructions, examples, context, and questions."
        ),
        "use_case": "Every LLM interaction starts with a prompt.",
        "examples": "'Summarize this document in 3 bullet points.'",
    },

    "System Prompt": {
        "explanation": (
            "A special prompt given to the LLM before the conversation starts. "
            "It sets the persona, rules, tone, and constraints for the model."
        ),
        "use_case": "Defining agent behavior, restricting topics, setting output format.",
        "examples": "'You are a helpful assistant. Always respond in JSON.'",
    },

    "Temperature": {
        "explanation": (
            "A parameter (0 to 2) controlling randomness in the model's output. "
            "Low = deterministic and focused. High = creative and varied. "
            "At 0, the model always picks the most likely token."
        ),
        "use_case": "Set low (0-0.3) for factual tasks, high (0.7-1.2) for creative writing.",
        "examples": "temp=0 for SQL generation, temp=1.0 for story writing.",
    },

    "Top-P (Nucleus Sampling)": {
        "explanation": (
            "Limits the model to sampling from the smallest set of tokens whose "
            "cumulative probability exceeds P. top_p=0.9 means only tokens covering "
            "90% of probability mass are considered."
        ),
        "use_case": "Alternative to temperature for controlling output diversity.",
        "examples": "top_p=0.1 = very focused, top_p=0.95 = more varied.",
    },

    "Top-K": {
        "explanation": (
            "Limits token sampling to only the K most likely next tokens at each step. "
            "Lower K = more deterministic."
        ),
        "use_case": "Used alongside temperature/top-p to control output quality.",
        "examples": "top_k=10 means only pick from the 10 most probable tokens.",
    },

    "Max Tokens": {
        "explanation": (
            "The maximum number of tokens the model can generate in its response. "
            "Setting this prevents runaway long outputs and controls cost."
        ),
        "use_case": "API calls, controlling response length.",
        "examples": "max_tokens=256 for short answers, 4096 for long documents.",
    },

    "Inference": {
        "explanation": (
            "The process of running a trained model to generate output. "
            "As opposed to training, inference uses the fixed weights of the model."
        ),
        "use_case": "Every time you call an LLM API, you are doing inference.",
        "examples": "Running Llama locally on an RTX 2050 is local inference.",
    },

    "Hallucination": {
        "explanation": (
            "When an LLM generates confident-sounding but factually incorrect or "
            "made-up information. The model does not 'know' it is wrong."
        ),
        "use_case": "Core problem RAG is designed to solve — grounding the model in real documents.",
        "examples": "Model invents a fake research paper citation.",
    },

    "Grounding": {
        "explanation": (
            "Connecting the LLM's response to verified, real-world information. "
            "Reduces hallucination by giving the model factual context to work from."
        ),
        "use_case": "RAG systems ground LLM answers in retrieved documents.",
        "examples": "Passing relevant chunks from a PDF before asking a question.",
    },

    "Fine-tuning": {
        "explanation": (
            "Further training a pretrained LLM on a smaller, domain-specific dataset "
            "to adapt it to a specific task or style."
        ),
        "use_case": "Medical QA, legal document analysis, coding assistants.",
        "examples": "Fine-tuning Llama on customer support conversations.",
    },

    "LoRA (Low-Rank Adaptation)": {
        "explanation": (
            "A parameter-efficient fine-tuning method that inserts small trainable "
            "matrices into the model layers instead of updating all weights. "
            "Much cheaper than full fine-tuning."
        ),
        "use_case": "Fine-tuning large models on consumer GPUs like your RTX 2050.",
        "examples": "Fine-tune a 7B model with 4GB VRAM using LoRA + 4-bit quantization.",
    },

    "QLoRA": {
        "explanation": (
            "LoRA applied to a quantized (4-bit) model. Dramatically reduces memory "
            "usage, making it possible to fine-tune large models on small GPUs."
        ),
        "use_case": "Fine-tuning 7B+ models on a single consumer GPU.",
        "examples": "Fine-tuning Llama 3.1 8B in 4-bit on an RTX 2050.",
    },

    "Quantization": {
        "explanation": (
            "Reducing the precision of model weights (e.g., from 32-bit float to "
            "4-bit integer) to shrink model size and speed up inference with minimal "
            "quality loss."
        ),
        "use_case": "Running large models on limited hardware.",
        "examples": "GGUF, GPTQ, AWQ, bitsandbytes 4-bit.",
    },

    "GGUF": {
        "explanation": (
            "A file format for storing quantized LLM weights, used by llama.cpp and "
            "Ollama for local inference. Successor to GGML."
        ),
        "use_case": "Running local LLMs efficiently on CPU or GPU.",
        "examples": "Download a GGUF file from HuggingFace and run with Ollama.",
    },

    "Embedding": {
        "explanation": (
            "A dense numerical vector representation of text that captures semantic "
            "meaning. Similar texts have embeddings that are close together in vector space."
        ),
        "use_case": "Core of RAG — documents are embedded and stored; queries are embedded to find similar chunks.",
        "examples": "text-embedding-ada-002, nomic-embed-text, sentence-transformers.",
    },

    "Attention Mechanism": {
        "explanation": (
            "The core operation in transformers. Allows each token to 'attend to' "
            "all other tokens in the sequence, learning which ones are most relevant."
        ),
        "use_case": "Foundation of how LLMs understand context and relationships between words.",
        "examples": "Self-attention in GPT. Cross-attention in encoder-decoder models.",
    },

    "Transformer": {
        "explanation": (
            "The neural network architecture behind almost all modern LLMs. "
            "Uses attention mechanisms to process sequences in parallel."
        ),
        "use_case": "Architecture of GPT, BERT, T5, Llama, Claude, and nearly all LLMs.",
        "examples": "'Attention Is All You Need' paper (Vaswani et al., 2017).",
    },

    "Tokenizer": {
        "explanation": (
            "Converts raw text into token IDs and back. Each model has its own "
            "tokenizer trained alongside it. BPE, WordPiece, and SentencePiece "
            "are common approaches."
        ),
        "use_case": "Preprocessing text before feeding to a model, decoding output tokens.",
        "examples": "tiktoken (OpenAI), tokenizers library (HuggingFace).",
    },

    "BPE (Byte Pair Encoding)": {
        "explanation": (
            "A subword tokenization algorithm. Starts with characters and iteratively "
            "merges the most frequent pairs. Balances vocabulary size with coverage."
        ),
        "use_case": "Default tokenizer strategy for GPT and many LLMs.",
        "examples": "GPT-4 uses BPE via tiktoken.",
    },

    "Prompt Engineering": {
        "explanation": (
            "The practice of designing effective prompts to get better outputs from "
            "LLMs without changing model weights. Includes techniques like few-shot, "
            "chain-of-thought, and role prompting."
        ),
        "use_case": "Improving LLM output quality without fine-tuning.",
        "examples": "'Think step by step' improves reasoning accuracy.",
    },

    "Few-Shot Prompting": {
        "explanation": (
            "Providing a few input-output examples in the prompt so the model "
            "learns the pattern and applies it to a new input."
        ),
        "use_case": "Teaching format, style, or task without fine-tuning.",
        "examples": "Show 3 examples of sentiment labels before asking model to classify.",
    },

    "Zero-Shot Prompting": {
        "explanation": (
            "Asking the model to perform a task with no examples — relying entirely "
            "on its pre-trained knowledge."
        ),
        "use_case": "Quick tasks where the model already knows the format.",
        "examples": "'Translate this to French: Hello' with no translation examples.",
    },

    "Chain-of-Thought (CoT)": {
        "explanation": (
            "Prompting technique that asks the model to reason step by step before "
            "giving a final answer. Dramatically improves accuracy on complex tasks."
        ),
        "use_case": "Math problems, logical reasoning, multi-step planning.",
        "examples": "'Think step by step before answering.' or just showing CoT examples.",
    },

    "RLHF (Reinforcement Learning from Human Feedback)": {
        "explanation": (
            "Training technique where human raters rank model outputs, and a reward "
            "model is trained on those preferences. The LLM is then fine-tuned using "
            "RL to maximize reward."
        ),
        "use_case": "Aligning LLMs to be helpful, harmless, and honest.",
        "examples": "Used to train ChatGPT, Claude, and Gemini.",
    },

    "DPO (Direct Preference Optimization)": {
        "explanation": (
            "Alternative to RLHF that skips the separate reward model. Directly "
            "optimizes the LLM on preference pairs (chosen vs rejected responses). "
            "Simpler and often more stable than RLHF."
        ),
        "use_case": "Aligning models to human preferences without RL complexity.",
        "examples": "Used in many open-source model fine-tunes.",
    },

    "KV Cache (Key-Value Cache)": {
        "explanation": (
            "During inference, attention keys and values for processed tokens are "
            "cached so they don't need to be recomputed for each new token. "
            "Dramatically speeds up generation."
        ),
        "use_case": "Makes autoregressive generation fast, especially for long contexts.",
        "examples": "PagedAttention in vLLM manages KV cache efficiently.",
    },

    "Speculative Decoding": {
        "explanation": (
            "A technique where a small draft model generates several tokens quickly, "
            "and the large model verifies them in parallel. Speeds up generation "
            "without changing output quality."
        ),
        "use_case": "Faster inference for large models.",
        "examples": "Using Llama 3.2 1B as draft model for Llama 3.1 8B.",
    },

    "Perplexity": {
        "explanation": (
            "A metric measuring how well an LLM predicts a text. Lower perplexity = "
            "model is less 'surprised' by the text = better fit. Used to evaluate "
            "language models."
        ),
        "use_case": "Comparing model quality, evaluating fine-tuned models.",
        "examples": "A perplexity of 5 is better than 20 on the same test set.",
    },

    "BLEU / ROUGE": {
        "explanation": (
            "Metrics for evaluating text generation. BLEU measures n-gram overlap "
            "between generated and reference text (used for translation). ROUGE "
            "measures recall of n-grams (used for summarization)."
        ),
        "use_case": "Automated evaluation of model output quality.",
        "examples": "ROUGE-L for summarization, BLEU-4 for translation benchmarks.",
    },

    "In-Context Learning (ICL)": {
        "explanation": (
            "The ability of LLMs to learn a task from examples provided in the "
            "prompt itself, without any gradient updates or weight changes."
        ),
        "use_case": "Adapting model behavior at inference time via examples in the prompt.",
        "examples": "Few-shot prompting is a form of ICL.",
    },

    "Instruction Tuning": {
        "explanation": (
            "Fine-tuning an LLM on (instruction, response) pairs so it learns to "
            "follow natural language instructions. Turns a raw language model into "
            "an assistant."
        ),
        "use_case": "Creating chat models from base models.",
        "examples": "Alpaca, FLAN-T5, Llama-chat variants.",
    },

    "Base Model vs Chat Model": {
        "explanation": (
            "Base model: raw pretrained LLM that just predicts next tokens. "
            "Chat model: base model fine-tuned with instruction tuning + RLHF to "
            "follow instructions and have conversations."
        ),
        "use_case": "Use chat models for assistants, base models for fine-tuning.",
        "examples": "Llama 3.1 (base) vs Llama 3.1-Instruct (chat).",
    },

    "Mixture of Experts (MoE)": {
        "explanation": (
            "Architecture where the model has multiple 'expert' sub-networks and a "
            "router that selects which experts to activate for each token. Only a "
            "fraction of parameters are active per forward pass."
        ),
        "use_case": "Scale model capacity without proportionally increasing compute.",
        "examples": "Mixtral 8x7B, GPT-4 (rumored MoE), Qwen MoE.",
    },

    "GQA (Grouped Query Attention)": {
        "explanation": (
            "Attention variant where multiple query heads share one key-value head. "
            "Reduces memory usage and speeds up inference compared to multi-head attention."
        ),
        "use_case": "Used in modern efficient LLMs to reduce KV cache size.",
        "examples": "Llama 3, Qwen2.5, Mistral use GQA.",
    },

    "RoPE (Rotary Position Embedding)": {
        "explanation": (
            "A position encoding scheme that encodes token position by rotating "
            "the query and key vectors. Handles longer sequences better than "
            "absolute position embeddings."
        ),
        "use_case": "Enables context length extension in modern LLMs.",
        "examples": "Used in Llama, Qwen, Mistral, and most modern LLMs.",
    },

    "SwiGLU": {
        "explanation": (
            "An activation function used in the feed-forward layers of modern LLMs. "
            "Performs better than ReLU and GELU in practice."
        ),
        "use_case": "Standard activation in Llama, PaLM, and most modern LLMs.",
        "examples": "Replaces the standard FFN in transformer blocks.",
    },

    "RMS Norm": {
        "explanation": (
            "A simplified layer normalization that only uses root mean square scaling "
            "without mean subtraction. Faster and simpler than LayerNorm."
        ),
        "use_case": "Pre-normalization in modern LLMs for training stability.",
        "examples": "Used in Llama, Qwen, Mistral instead of standard LayerNorm.",
    },

    "vLLM": {
        "explanation": (
            "A high-performance inference engine for LLMs. Uses PagedAttention to "
            "manage KV cache efficiently, enabling high throughput serving."
        ),
        "use_case": "Serving LLMs at scale with high throughput.",
        "examples": "Deploy Llama 3 with vLLM for production API serving.",
    },

    "Ollama": {
        "explanation": (
            "A tool for running LLMs locally. Handles model download, quantization, "
            "and serving with a simple API compatible with OpenAI."
        ),
        "use_case": "Local LLM inference for development and privacy.",
        "examples": "ollama run llama3.1 or ollama pull qwen2.5",
    },

}
RAG_TERMS = {

    "RAG (Retrieval Augmented Generation)": {
        "explanation": (
            "A technique that enhances LLM responses by first retrieving relevant "
            "documents from a knowledge base, then passing them as context to the LLM. "
            "Reduces hallucination and keeps knowledge up to date."
        ),
        "use_case": "Document Q&A, enterprise knowledge bases, customer support bots.",
        "examples": "Ask a question → retrieve relevant PDF chunks → LLM answers using chunks.",
    },

    "Vector Database": {
        "explanation": (
            "A database optimized for storing and searching high-dimensional embedding "
            "vectors. Supports approximate nearest neighbor (ANN) search to find "
            "semantically similar documents quickly."
        ),
        "use_case": "Storing document embeddings for RAG retrieval.",
        "examples": "FAISS, Chroma, Pinecone, Weaviate, Qdrant, Milvus.",
    },

    "FAISS (Facebook AI Similarity Search)": {
        "explanation": (
            "An open-source library for efficient similarity search over dense vectors. "
            "Supports multiple index types (Flat, IVF, HNSW) with different "
            "speed/accuracy trade-offs."
        ),
        "use_case": "Local vector search without needing a server.",
        "examples": "IndexFlatL2 for exact search, IndexIVFFlat for approximate.",
    },

    "Chunking": {
        "explanation": (
            "Breaking large documents into smaller pieces before embedding and storing. "
            "Chunk size and overlap are critical hyperparameters that affect retrieval quality."
        ),
        "use_case": "Preparing documents for RAG. Affects what context gets retrieved.",
        "examples": "Split a 50-page PDF into 512-token chunks with 50-token overlap.",
    },

    "Chunk Size": {
        "explanation": (
            "The number of tokens (or characters) in each chunk. Too small = loses "
            "context. Too large = retrieves too much noise. Typically 256–1024 tokens."
        ),
        "use_case": "Tuning retrieval precision vs context richness.",
        "examples": "Code = large chunks. FAQs = small chunks.",
    },

    "Chunk Overlap": {
        "explanation": (
            "How many tokens adjacent chunks share. Overlap ensures information at "
            "chunk boundaries is not lost during retrieval."
        ),
        "use_case": "Prevents answers from being split across chunk boundaries.",
        "examples": "Chunk size=512, overlap=50 — last 50 tokens of chunk N are first 50 of chunk N+1.",
    },

    "Similarity Search": {
        "explanation": (
            "Finding the most semantically similar stored chunks to a query. "
            "Done by computing distance between query embedding and stored embeddings. "
            "Common metrics: cosine similarity, L2 (Euclidean), dot product."
        ),
        "use_case": "Core retrieval step in RAG.",
        "examples": "Query: 'What is the refund policy?' → retrieve top-3 most similar chunks.",
    },

    "Cosine Similarity": {
        "explanation": (
            "A metric measuring the angle between two vectors. Values range from -1 "
            "to 1, where 1 = identical direction = most similar. Used to compare embeddings."
        ),
        "use_case": "Ranking retrieved documents by relevance in RAG.",
        "examples": "cos_sim(query_emb, doc_emb) > 0.8 = highly relevant.",
    },

    "Dense Retrieval": {
        "explanation": (
            "Retrieval using embedding vectors (neural). Captures semantic meaning "
            "and can find relevant documents even if they use different words than the query."
        ),
        "use_case": "Default retrieval method in most RAG systems.",
        "examples": "Embed query and docs with sentence-transformers, search with FAISS.",
    },

    "Sparse Retrieval (BM25)": {
        "explanation": (
            "Keyword-based retrieval using term frequency statistics. Fast and "
            "effective for exact keyword matches. BM25 is the standard algorithm."
        ),
        "use_case": "When exact terms matter — product codes, names, technical terms.",
        "examples": "BM25 with rank_bm25 library, Elasticsearch.",
    },

    "Hybrid Retrieval": {
        "explanation": (
            "Combines dense (semantic) and sparse (keyword) retrieval. Gets the best "
            "of both — semantic understanding + exact keyword matching. "
            "Scores are fused (e.g. RRF or weighted sum)."
        ),
        "use_case": "Production RAG systems for best retrieval accuracy.",
        "examples": "FAISS (dense) + BM25 (sparse) → fuse scores → rerank.",
    },

    "RRF (Reciprocal Rank Fusion)": {
        "explanation": (
            "A score fusion method for hybrid retrieval. Combines rankings from "
            "multiple retrieval systems by summing reciprocals of each document's "
            "rank in each system."
        ),
        "use_case": "Merging dense and sparse retrieval rankings in hybrid RAG.",
        "examples": "RRF(doc) = 1/(k+rank_dense) + 1/(k+rank_sparse)",
    },

    "Reranking": {
        "explanation": (
            "A second pass after initial retrieval that uses a more powerful model "
            "(cross-encoder) to reorder retrieved chunks by true relevance. "
            "More accurate than bi-encoder retrieval but slower."
        ),
        "use_case": "Improving retrieval precision — retrieve 20, rerank to top 5.",
        "examples": "Cross-encoders, Cohere Rerank, FlashRank, BGE reranker.",
    },

    "Cross-Encoder": {
        "explanation": (
            "A model that takes a query and document together and scores their relevance "
            "jointly. More accurate than bi-encoders but cannot pre-compute embeddings "
            "— must be run at query time."
        ),
        "use_case": "Reranking in RAG, semantic similarity scoring.",
        "examples": "ms-marco-MiniLM-L-6-v2, BGE-reranker.",
    },

    "Bi-Encoder": {
        "explanation": (
            "A model that encodes query and document separately into vectors. "
            "Fast because document embeddings can be precomputed. "
            "Less accurate than cross-encoders for fine-grained relevance."
        ),
        "use_case": "First-stage retrieval in RAG systems.",
        "examples": "sentence-transformers, nomic-embed-text, text-embedding-3-small.",
    },

    "HNSW (Hierarchical Navigable Small World)": {
        "explanation": (
            "A graph-based approximate nearest neighbor index. Very fast at query "
            "time, good recall, but uses more memory than IVF indexes."
        ),
        "use_case": "Fast vector search in production RAG systems.",
        "examples": "Default index in Qdrant, Weaviate, and supported in FAISS.",
    },

    "IVF (Inverted File Index)": {
        "explanation": (
            "A FAISS index type that clusters vectors into Voronoi cells. "
            "At query time, only nearby cells are searched. Faster than flat "
            "search but requires training."
        ),
        "use_case": "Scalable approximate nearest neighbor search for large datasets.",
        "examples": "faiss.IndexIVFFlat, faiss.IndexIVFPQ",
    },

    "Document Loader": {
        "explanation": (
            "A component that reads raw documents from various sources and converts "
            "them to a standard format for processing."
        ),
        "use_case": "Ingesting PDFs, Word docs, web pages, databases into a RAG pipeline.",
        "examples": "LangChain loaders: PyPDFLoader, WebBaseLoader, CSVLoader.",
    },

    "Text Splitter": {
        "explanation": (
            "Splits raw text into chunks for embedding. Can split by character count, "
            "token count, sentence, paragraph, or recursively by multiple separators."
        ),
        "use_case": "Chunking step in RAG ingestion pipeline.",
        "examples": "RecursiveCharacterTextSplitter, TokenTextSplitter in LangChain.",
    },

    "Ingestion Pipeline": {
        "explanation": (
            "The offline process of loading documents, splitting into chunks, "
            "embedding each chunk, and storing in a vector database."
        ),
        "use_case": "Building the knowledge base for a RAG system.",
        "examples": "Load PDF → split → embed with nomic → store in FAISS.",
    },

    "Retrieval Pipeline": {
        "explanation": (
            "The online process at query time: embed the query, search the vector "
            "database, retrieve top-k chunks, pass to LLM as context."
        ),
        "use_case": "Answering user queries in a RAG system.",
        "examples": "Query → embed → FAISS search → top 5 chunks → LLM → answer.",
    },

    "Top-K Retrieval": {
        "explanation": (
            "Retrieving the K most similar chunks to the query. K is a hyperparameter "
            "— too low misses context, too high adds noise."
        ),
        "use_case": "Controlling how many chunks go into the LLM context.",
        "examples": "k=5 for short context models, k=20 for long context models.",
    },

    "Metadata Filtering": {
        "explanation": (
            "Filtering retrieved chunks based on metadata (date, author, source, "
            "category) before or after similarity search. Narrows search space."
        ),
        "use_case": "Multi-tenant RAG, date-filtered search, department-specific knowledge bases.",
        "examples": "filter={'source': 'finance_docs'} before vector search.",
    },

    "Contextual Compression": {
        "explanation": (
            "Compressing or extracting only the relevant portion of a retrieved chunk "
            "before passing to the LLM. Reduces noise and saves context window space."
        ),
        "use_case": "When chunks are large and only a sentence or two is relevant.",
        "examples": "LangChain ContextualCompressionRetriever.",
    },

    "Multi-Query Retrieval": {
        "explanation": (
            "Using the LLM to generate multiple versions or sub-questions from the "
            "original query, then retrieving for each. Improves recall by covering "
            "more semantic angles."
        ),
        "use_case": "When a query is ambiguous or complex.",
        "examples": "'What are the benefits of X?' → also query 'advantages of X', 'why use X'.",
    },

    "HyDE (Hypothetical Document Embedding)": {
        "explanation": (
            "Ask the LLM to generate a hypothetical answer to the query, then embed "
            "that answer instead of the raw query. The hypothesis is often closer "
            "in embedding space to real documents."
        ),
        "use_case": "Improving retrieval for complex or abstract queries.",
        "examples": "Query → LLM generates fake answer → embed fake answer → search.",
    },

    "Parent-Child Chunking": {
        "explanation": (
            "Store small child chunks for precise retrieval, but return their larger "
            "parent chunks to the LLM for richer context."
        ),
        "use_case": "Balance retrieval precision with generation context richness.",
        "examples": "Child = 128 tokens (retrieved), Parent = 512 tokens (sent to LLM).",
    },

    "Semantic Chunking": {
        "explanation": (
            "Splitting documents at semantically meaningful boundaries instead of "
            "fixed token counts. Uses embedding similarity to detect topic shifts."
        ),
        "use_case": "Better chunk coherence — each chunk covers one idea.",
        "examples": "LangChain SemanticChunker.",
    },

    "MMR (Maximal Marginal Relevance)": {
        "explanation": (
            "A retrieval strategy that balances relevance and diversity. Avoids "
            "returning near-duplicate chunks by penalizing similarity to already "
            "selected results."
        ),
        "use_case": "When top-k retrieval returns redundant chunks.",
        "examples": "FAISS MMR search or LangChain retriever with search_type='mmr'.",
    },

    "Self-RAG": {
        "explanation": (
            "A RAG variant where the LLM itself decides when to retrieve, what to "
            "retrieve, and whether the retrieved content is relevant. The model "
            "generates special reflection tokens."
        ),
        "use_case": "Adaptive RAG that avoids unnecessary retrieval.",
        "examples": "Self-RAG paper (Asai et al., 2023).",
    },

    "Corrective RAG (CRAG)": {
        "explanation": (
            "Evaluates the quality of retrieved documents and falls back to web search "
            "or other sources if retrieved docs are not relevant enough."
        ),
        "use_case": "Robust RAG that handles retrieval failures gracefully.",
        "examples": "CRAG paper (Shi et al., 2024) — implemented with LangGraph.",
    },

    "Agentic RAG": {
        "explanation": (
            "RAG where an agent decides the retrieval strategy — what to query, "
            "how many times, from which source. Can iteratively refine until satisfied."
        ),
        "use_case": "Complex multi-step research tasks, multi-source knowledge bases.",
        "examples": "LangGraph agent that calls FAISS, then web search, then combines.",
    },

    "LLM Fallback": {
        "explanation": (
            "When retrieved documents do not contain enough information, fall back "
            "to the LLM's pretrained knowledge to answer."
        ),
        "use_case": "Handling queries outside the knowledge base gracefully.",
        "examples": "if hybrid_score < threshold: use LLM alone, else use RAG.",
    },

    "Knowledge Graph RAG": {
        "explanation": (
            "Augmenting RAG with a knowledge graph that represents entities and "
            "relationships. Enables structured reasoning over retrieved information."
        ),
        "use_case": "When relationships between entities matter (org charts, medical ontologies).",
        "examples": "GraphRAG by Microsoft, Neo4j + LLM.",
    },

}
AGENT_TERMS = {

    "AI Agent": {
        "explanation": (
            "An LLM-powered system that perceives its environment, makes decisions, "
            "takes actions (using tools), and works toward a goal over multiple steps. "
            "Unlike a single LLM call, agents operate in a loop."
        ),
        "use_case": "Automating complex multi-step tasks that require planning and tool use.",
        "examples": "Browser agent, coding agent, research agent, file manager agent.",
    },

    "ReAct (Reasoning + Acting)": {
        "explanation": (
            "A prompting framework where the agent alternates between Thought "
            "(reasoning about what to do) and Action (calling a tool), followed "
            "by Observation (seeing the result). Loops until done."
        ),
        "use_case": "Standard agent loop for most tool-using LLM agents.",
        "examples": "Thought: I need to search → Action: search('query') → Observation: result → ...",
    },

    "Tool / Function Calling": {
        "explanation": (
            "The ability for an LLM to call external functions or APIs. The model "
            "outputs a structured JSON specifying which tool to call and with what "
            "arguments. The framework executes it and returns the result."
        ),
        "use_case": "Giving agents abilities: search, code execution, file I/O, API calls.",
        "examples": "OpenAI function calling, LangChain tools, NVIDIA NIM tool use.",
    },

    "Tool Schema": {
        "explanation": (
            "A JSON description of a tool's name, description, and parameters. "
            "The LLM uses this to understand when and how to call the tool."
        ),
        "use_case": "Defining tools for the LLM to choose from.",
        "examples": "{'name': 'search', 'description': 'Search the web', 'parameters': {...}}",
    },

    "Agent Loop": {
        "explanation": (
            "The repeated cycle of: receive input → reason → choose action → "
            "execute tool → observe result → repeat until goal is met or done."
        ),
        "use_case": "Core execution pattern of all LLM agents.",
        "examples": "While not done: action = llm(state) → result = execute(action) → state.update(result)",
    },

    "Planning": {
        "explanation": (
            "The agent's ability to break a complex goal into a sequence of sub-tasks "
            "and execute them in order. Can be done upfront (plan-then-execute) or "
            "dynamically (plan as you go)."
        ),
        "use_case": "Research tasks, multi-step automation, complex problem solving.",
        "examples": "Plan-and-Execute agent: first make a plan, then execute each step.",
    },

    "Memory (Agent)": {
        "explanation": (
            "How agents persist and access information across steps. Types: "
            "In-context (conversation history), External (vector DB), "
            "Episodic (past interactions), Semantic (facts), Procedural (skills)."
        ),
        "use_case": "Maintaining context over long tasks, recalling past information.",
        "examples": "Store tool results in a list passed back to LLM each step.",
    },

    "Short-Term Memory": {
        "explanation": (
            "Information available within the current agent run — typically the "
            "conversation history and tool outputs stored in the context window."
        ),
        "use_case": "Keeping track of what happened in the current task.",
        "examples": "history list passed to LLM at each step.",
    },

    "Long-Term Memory": {
        "explanation": (
            "Information persisted across agent runs — stored externally in a vector "
            "database, key-value store, or database."
        ),
        "use_case": "Remembering user preferences, past conversations, accumulated knowledge.",
        "examples": "Store key facts in FAISS, retrieve at start of each new session.",
    },

    "Observation": {
        "explanation": (
            "The result returned to the agent after executing an action/tool. "
            "The agent uses observations to decide what to do next."
        ),
        "use_case": "Feeding tool results back to the LLM in the agent loop.",
        "examples": "Tool result: 'File created successfully at /path/file.txt'",
    },

    "Action Space": {
        "explanation": (
            "The set of all possible actions an agent can take. Defines what the "
            "agent is capable of doing — its available tools."
        ),
        "use_case": "Designing what capabilities an agent has.",
        "examples": "['read_file', 'write_file', 'search_web', 'run_code', 'done']",
    },

    "Multi-Agent System": {
        "explanation": (
            "A system where multiple specialized agents collaborate to complete a task. "
            "An orchestrator agent delegates subtasks to worker agents."
        ),
        "use_case": "Complex tasks requiring different expertise — research + coding + writing.",
        "examples": "Orchestrator → Research Agent, Coding Agent, Writing Agent.",
    },

    "Orchestrator": {
        "explanation": (
            "The main agent that receives the high-level goal, creates a plan, "
            "and delegates subtasks to specialized worker agents."
        ),
        "use_case": "Managing multi-agent workflows.",
        "examples": "Boss agent that assigns tasks to researcher, coder, and reviewer agents.",
    },

    "Worker Agent": {
        "explanation": (
            "A specialized agent that receives a specific subtask from the orchestrator "
            "and executes it using its specialized tools."
        ),
        "use_case": "Handling specific parts of a larger pipeline.",
        "examples": "Code agent, web search agent, data analysis agent.",
    },

    "LangChain": {
        "explanation": (
            "A Python framework for building LLM applications. Provides abstractions "
            "for chains, agents, tools, memory, and retrievers. Most popular "
            "LLM framework."
        ),
        "use_case": "Building RAG pipelines, agents, chatbots.",
        "examples": "LangChain chains, LCEL, LangChain agents.",
    },

    "LangGraph": {
        "explanation": (
            "A library built on top of LangChain for building stateful, multi-actor "
            "agent workflows as graphs. Nodes = agent steps, Edges = transitions. "
            "Supports cycles, branching, and human-in-the-loop."
        ),
        "use_case": "Complex multi-step agent workflows, multi-agent systems.",
        "examples": "Build a research agent as a graph: search → summarize → write → review.",
    },

    "MCP (Model Context Protocol)": {
        "explanation": (
            "An open protocol by Anthropic that standardizes how LLMs connect to "
            "external tools and data sources. Defines a standard interface between "
            "agents and tool servers."
        ),
        "use_case": "Building interoperable agent tool ecosystems.",
        "examples": "Claude connecting to Google Drive, Slack, or custom tools via MCP.",
    },

    "Human-in-the-Loop": {
        "explanation": (
            "A design pattern where the agent pauses and asks a human to approve "
            "or correct an action before executing it. Critical for high-stakes actions."
        ),
        "use_case": "File deletion, financial transactions, sending emails.",
        "examples": "Agent shows planned action → human approves → agent executes.",
    },

    "Guardrails": {
        "explanation": (
            "Safety constraints that prevent agents from taking harmful or unintended "
            "actions. Can be input filters, output filters, or action validators."
        ),
        "use_case": "Preventing agents from deleting critical files, leaking data, or going rogue.",
        "examples": "Whitelist of allowed directories, max step limits, blocked actions.",
    },

    "Tree of Thoughts (ToT)": {
        "explanation": (
            "A planning method where the agent explores multiple reasoning paths "
            "as a tree, evaluates branches, and backtracks if needed. Better than "
            "linear chain-of-thought for complex problems."
        ),
        "use_case": "Complex multi-step reasoning where the first approach may fail.",
        "examples": "Solving puzzles, strategic planning, code debugging.",
    },

    "Reflexion": {
        "explanation": (
            "An agent framework where the agent reflects on its failures and generates "
            "verbal feedback to itself, then retries. Improves over multiple attempts "
            "without gradient updates."
        ),
        "use_case": "Self-improving agents that learn from mistakes in context.",
        "examples": "Agent fails a task → reflects → generates improvement strategy → retries.",
    },

    "Plan-and-Execute": {
        "explanation": (
            "A two-phase agent pattern: first generate a complete plan (list of steps), "
            "then execute each step in order. More predictable than pure ReAct."
        ),
        "use_case": "Tasks where the full plan is known upfront.",
        "examples": "Plan: [search, summarize, write, save] → execute each step.",
    },

    "Subagent": {
        "explanation": (
            "An agent called by another agent to handle a specific subtask. "
            "The calling agent treats the subagent as a tool."
        ),
        "use_case": "Modular agent design — each subagent handles one domain.",
        "examples": "Main agent calls a 'web search subagent' and a 'code execution subagent'.",
    },

    "Tool Use": {
        "explanation": (
            "The ability of an LLM agent to call external tools — APIs, databases, "
            "code runners, file systems, web browsers — to extend its capabilities "
            "beyond text generation."
        ),
        "use_case": "Any real-world agent task — search, file I/O, database queries.",
        "examples": "search_web(), run_python(), read_file(), send_email().",
    },

    "Code Execution (Agent)": {
        "explanation": (
            "An agent capability where the LLM writes code (Python, bash, etc.) and "
            "executes it in a sandbox, observing the output. Enables data analysis, "
            "file manipulation, web scraping."
        ),
        "use_case": "Data science agents, automation agents, debugging agents.",
        "examples": "OpenAI Code Interpreter, E2B sandbox, local Python exec.",
    },

    "Browser Agent": {
        "explanation": (
            "An agent that controls a real web browser (via Playwright, Selenium, or "
            "Puppeteer) to navigate websites, click, type, and extract information."
        ),
        "use_case": "Web automation, scraping dynamic sites, filling forms, booking.",
        "examples": "Your Playwright + Llama 3.1 browser agent project.",
    },

    "Accessibility Tree": {
        "explanation": (
            "A structured representation of a web page's interactive elements — "
            "buttons, inputs, links, text — in a hierarchical tree format. "
            "Used by browser agents as their observation instead of raw HTML or screenshots."
        ),
        "use_case": "Giving browser agents a compact, actionable view of the page.",
        "examples": "page.accessibility.snapshot() in Playwright.",
    },

    "Agentic Loop": {
        "explanation": (
            "Same as agent loop — the continuous perceive → think → act → observe "
            "cycle that an agent runs until the task is complete."
        ),
        "use_case": "Foundation of all autonomous agent systems.",
        "examples": "while not done: action = llm(obs) → obs = env.step(action)",
    },

    "State (Agent)": {
        "explanation": (
            "All information the agent holds about the current task — goal, history, "
            "intermediate results, current step, and pending actions."
        ),
        "use_case": "Passing context between agent steps so each step knows what happened before.",
        "examples": "{'goal': ..., 'history': [...], 'step': 3, 'pending': None}",
    },

    "Structured Output": {
        "explanation": (
            "Forcing the LLM to output in a specific format (JSON, XML) rather than "
            "free text. Used to reliably parse the model's action choices."
        ),
        "use_case": "Parsing agent actions reliably without regex hacks.",
        "examples": "{'action': 'read_file', 'args': {'path': '/tmp/data.txt'}}",
    },

    "Hallucination (Agent)": {
        "explanation": (
            "When an agent invents tool calls, file paths, URLs, or results that do "
            "not exist. More dangerous than chat hallucination because it can cause "
            "real-world actions."
        ),
        "use_case": "Critical failure mode to guard against in production agents.",
        "examples": "Agent calls delete_file('/important/system/file') on a nonexistent path.",
    },

    "Max Steps / Step Limit": {
        "explanation": (
            "A safety limit on how many steps an agent can take before stopping. "
            "Prevents infinite loops and runaway agents."
        ),
        "use_case": "Safety and cost control in agent systems.",
        "examples": "if state['step'] > 20: stop and return current result.",
    },

    "NVIDIA NIM": {
        "explanation": (
            "NVIDIA's inference microservices — optimized, containerized LLM APIs "
            "compatible with the OpenAI API format. Supports Llama, Mistral, and other models."
        ),
        "use_case": "Running LLMs via API with NVIDIA-optimized performance.",
        "examples": "Your browser agent uses NVIDIA NIM with Llama 3.1 8B.",
    },

}
