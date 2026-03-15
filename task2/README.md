# Repo-Context Retriever: "Best Strategy"

A high-precision context retrieval pipeline designed to identify and rank the most relevant code snippets across a repository to assist AI code generation.

---

## 🚀 Core Features
* **Multi-Signal Ranking:** Combines import graphs, structural proximity, and content overlap.
* **Syntactic Awareness:** Automatically identifies the current code block (function/class) to build better search queries.
* **Semantic Search:** Uses `all-MiniLM-L6-v2` embeddings to find relevant code chunks within files.
* **Token Efficiency:** Greedily fills a token budget (default: 6,000) with the highest-scoring snippets.

---

## 🛠️ How It Works

### 1. File Selection (Three-Tier Priority)
The system scores every file in the repository using a tiered approach:
* **Tier 3 (Recent):** Files recently modified by the user.
* **Tier 2 (Imports):** Local files explicitly imported in the current script.
* **Tier 1 (Similarity):** Other files ranked by **Lines IoU** (content overlap) multiplied by **Path Proximity** (structural distance).

### 2. Intelligent Querying
Instead of using raw text, the retriever cleans the search query:
* **Scope Extraction:** Finds the "strong" syntactic block (e.g., `def`, `class`, `for`, `while`) the cursor is currently in.
* **Noise Reduction:** Strips library imports and "empty" suffixes (trailing brackets/semicolons) that don't help retrieval.

### 3. Chunking & Scoring
* **Windowing:** Files are broken into overlapping chunks (default: 30 lines).
* **Similarity:** The top chunks per file are selected based on cosine similarity to the intent query.
* **Selection:** Chunks are globally ranked and selected until the **token budget** is reached.

### 4. Context Assembly
The final context is formatted as a single string:
* **Ordering:** Chunks are ordered from **least relevant to most relevant** (most relevant file placed last).
* **Separator:** Files are demarcated by a `<|file_sep|>` token.
* **Formatting:** Includes the relative file name and specific excerpts separated by ellipses (`...`).

---

## 💻 Usage

Run the script via command line by specifying the language and project stage:

```bash
poetry run python final_idea.py --lang python --stage practice --max-files 8 --token-budget 16000
