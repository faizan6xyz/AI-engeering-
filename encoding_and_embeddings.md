# Types of Encoding & Embeddings

These are **2 different concepts** but closely related. Here's a complete breakdown:

---

## 🔤 PART 1: Types of ENCODING

Encoding = **converting data into a different format/representation**

---

### 1. 📝 Text Encoding

| Type | What it does | Example |
|---|---|---|
| **ASCII** | Basic English characters | A=65, B=66 |
| **UTF-8** | All world languages | Supports Hindi, Arabic, Chinese |
| **UTF-16** | Unicode with 2 bytes | Used in Windows |
| **Base64** | Binary to text | Images in emails |
| **URL Encoding** | Special chars in URLs | Space = %20 |

---

### 2. 🤖 ML/AI Encoding (Most Relevant to LLMs)

| Type | What it does | Used In |
|---|---|---|
| **One-Hot Encoding** | Each category = binary array | Basic ML models |
| **Label Encoding** | Each category = a number | Decision trees |
| **Tokenization** | Text → token IDs | All LLMs |
| **Positional Encoding** | Adds position info to tokens | Transformers |
| **BPE (Byte Pair Encoding)** | Splits words into subwords | GPT, Claude |
| **WordPiece** | Similar to BPE | BERT, Google |
| **SentencePiece** | Language-independent tokenizer | T5, LLaMA |

---

### 3. 🖼️ Media Encoding

| Type | What it does |
|---|---|
| **Image Encoding** | Pixels → numbers (RGB) |
| **Audio Encoding** | Sound waves → numbers |
| **Video Encoding** | Frames + audio → compressed format |
| **Binary Encoding** | Data → 0s and 1s |

---

### 4. 🔐 Cryptographic Encoding

| Type | What it does |
|---|---|
| **MD5** | Hashes data into fixed string |
| **SHA-256** | Secure hashing |
| **RSA** | Public/private key encoding |
| **AES** | Symmetric encryption |

---

## 🧠 PART 2: Types of EMBEDDINGS

Embedding = **converting data into a dense vector of numbers** that captures MEANING

> Unlike encoding (just converts format), embeddings **capture semantic meaning**

---

### 1. 📝 Word Embeddings (Classic)

| Type | Year | Key Feature |
|---|---|---|
| **Word2Vec** | 2013 | Words with similar meaning are close together |
| **GloVe** | 2014 | Uses global word co-occurrence |
| **FastText** | 2016 | Works on subwords, handles typos |
| **ELMo** | 2018 | Context-aware word vectors |

**Example:**
```
King - Man + Woman = Queen
(embeddings capture this relationship!)
```

---

### 2. 📄 Sentence & Document Embeddings

| Type | What it does |
|---|---|
| **Sentence-BERT** | Whole sentence → single vector |
| **Doc2Vec** | Entire document → vector |
| **Universal Sentence Encoder** | Google's sentence embeddings |
| **InferSent** | Facebook's sentence embeddings |

---

### 3. 🤖 Transformer Embeddings (Modern)

| Type | Model | Special Feature |
|---|---|---|
| **BERT Embeddings** | BERT | Bidirectional context |
| **GPT Embeddings** | GPT-4 | Great for generation tasks |
| **Claude Embeddings** | Claude | Anthropic's embeddings |
| **OpenAI Embeddings** | text-embedding-3 | Most popular for RAG |
| **Cohere Embeddings** | Cohere | Multilingual support |

---

### 4. 🖼️ Multimodal Embeddings

| Type | What it embeds |
|---|---|
| **CLIP** | Images + Text together |
| **DALL-E Embeddings** | Image generation vectors |
| **ImageBind** | Image, Audio, Text, Video |
| **Flamingo** | Visual + Language combined |

---

### 5. 🔍 Specialized Embeddings

| Type | Used For |
|---|---|
| **Graph Embeddings** | Social networks, knowledge graphs |
| **Code Embeddings** | GitHub Copilot, code search |
| **Audio Embeddings** | Speech recognition, music |
| **Video Embeddings** | Video search, classification |
| **Knowledge Graph Embeddings** | TransE, RotatE — for facts/relations |

---

## 📊 Encoding vs Embedding — Key Difference

| | Encoding | Embedding |
|---|---|---|
| **Purpose** | Format conversion | Meaning capture |
| **Output** | Any format | Dense number vector |
| **Captures meaning** | ❌ No | ✅ Yes |
| **Size** | Fixed or variable | Fixed dense vector |
| **Example** | A → 65 (ASCII) | "King" → [0.2, 0.8, 0.1...] |
| **Used in** | Data storage, transmission | AI, search, RAG |

---

## 🔄 How They Work Together in LLMs

```
Raw Text ("Hello World")
        ↓
   ENCODING (Tokenization)
   "Hello" → [15496]
   "World" → [2159]
        ↓
   EMBEDDING (Meaning)
   [15496] → [0.2, 0.8, 0.5, 0.1...]
   [2159]  → [0.9, 0.1, 0.3, 0.7...]
        ↓
   LLM processes the vectors
        ↓
   Output generated ✅
```

---

## 🌟 Key Takeaway

| | One Line Summary |
|---|---|
| **Encoding** | Converts data into another **format** |
| **Embedding** | Converts data into **meaningful numbers** |
| **Together** | Power every modern LLM and AI system |
