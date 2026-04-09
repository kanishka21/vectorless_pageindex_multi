📄 PageIndex Vectorless RAG System

A hierarchical, reasoning-based RAG system inspired by PageIndex that:

❌ Uses no vector database
❌ Uses no embeddings
✅ Uses LLM reasoning for retrieval
✅ Builds a tree-structured index (Table of Contents)
✅ Performs agentic tree traversal for answering queries
🧠 Architecture
PDFs
 ↓
Corpus Builder (TOC Tree)
 ↓
JSON Corpus Store
 ↓
Query
 ↓
LLM Tree Traversal (Reasoning)
 ↓
Relevant Section
 ↓
LLM Answer
🚀 Features
📚 Multi-document support
🌳 Hierarchical indexing (Section → Subsection)
🧠 LLM-based reasoning retrieval (no similarity search)
🔍 Explainable retrieval (visible reasoning path)
⚡ No vector DB / embedding dependency
📂 Project Structure
project/
 ├── build_corpus.py        # Builds tree index from PDFs
 ├── query_system.py        # Query + reasoning + answer
 ├── pdfs/                  # Input PDFs
 ├── corpus/                # Generated JSON corpus
⚙️ Setup
1. Install dependencies
pip install pypdf openai
2. Configure LLM endpoint

Open both files and update:

client = OpenAI(
    base_url="YOUR_INTERNAL_OPENAI_URL",
    api_key="YOUR_API_KEY"
)

Example:

base_url="http://localhost:8000/v1"
api_key="dummy"
📄 Step 1: Add PDFs

Place your documents inside:

pdfs/
 ├── file1.pdf
 ├── file2.pdf
🏗️ Step 2: Build Corpus

Run:

python build_corpus.py
What happens:
Reads PDFs
Generates Table of Contents using LLM
Builds hierarchical structure
Saves JSON files
Output:
corpus/
 ├── file1.json
 ├── file2.json
🔍 Step 3: Run Query System
python query_system.py
💬 Step 4: Ask Questions

Example:

Ask: What is the refund policy?
Ask: Explain termination clause
Ask: What are payment conditions?
🧠 How Retrieval Works

Unlike traditional RAG:

❌ Traditional RAG
Query → Vector Search → Retrieve
✅ This System
Query → LLM selects section → goes deeper → finds best node → answers
📊 Example Output
Path:
- Refund Policy
- Eligibility

Answer:
Customers can request a refund within 30 days...
⏱️ Performance
Stage	Time
Corpus build	2–5 sec / PDF
Tree traversal	2–4 sec
Answer generation	1–2 sec
Total query time	~3–6 sec
⚠️ Common Issues & Fixes
1. Empty or poor answers

✔ Ensure document text is included in corpus
✔ Use structured PDFs with headings

2. Slow execution

✔ Expected due to LLM reasoning
✔ Avoid rebuilding corpus repeatedly

3. JSON parsing errors

✔ Reduce very large PDFs
✔ Ensure clean text extraction

⚡ Optimization Tips
Cache corpus (already implemented)
Use smaller model for traversal
Use stronger model only for final answer
Limit document size if very large
🧠 When to Use This

Best for:

📜 Contracts
📊 Financial documents
📚 Research papers
📑 Policy documents
❌ Not Ideal For
Simple keyword lookup
Real-time (<1 sec latency) systems
Extremely large document collections
💡 Key Insight

This system is not just retrieval:

It simulates how a human reads and navigates documents

🚀 Future Improvements
🔥 Hybrid search (BM25 + PageIndex)
🔥 Beam search (multi-path reasoning)
🔥 FastAPI backend
🔥 Chat UI
🔥 Caching LLM responses
🧾 License

Use freely for learning and experimentation.
