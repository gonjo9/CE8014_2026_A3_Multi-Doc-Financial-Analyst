**You can edit functions in `graph_agent.py`, and in evaluator.py can change test_mode (LEGACY is langchain mode, GRAPH is langgraph)**  

# 🛠️ Prerequisites
Before you begin, ensure you have the following installed:

* Python 3.11 (Strict requirement) 

* Google Cloud API Key or other LLM Key
# ⚙️ Environment Setup
### 1. Virtual Environment Setup

It is highly recommended to use a virtual environment to manage dependencies.

**For macOS / Linux:**
```
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate
```
**For Windows:**
```
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate
```

### 2. Install Dependencies

`pip install -r requirements.txt`

### 3. Environment Variables (.env)

Rename the file `.env_example` to `.env` in the root directory and add your API_KEY

# 📂 File Descriptions

* **data/:** Folder containing the raw PDF financial reports
* **langgraph_agent.py:** [MAIN WORKSPACE] This is where you will write your code. It contains the logic for:
  * PDF Ingestion: `initialize_vector_dbs()`

  *  Graph Nodes: `retrieve_node`, `grade_documents_node`, `generate_node`, `rewrite_node`.

  *  Legacy Agent: `run_legacy_agent` (The baseline for comparison).
* **evaluator.py:** The benchmark testing script. It runs a suite of test cases (Apple Revenue, Tesla R&D, Comparison, Traps) and uses "LLM-as-a-Judge" to score your agent (Pass/Fail).
* **config.py:** Configuration file that handles API key loading and initializes the LLM and Embedding models.


# 📝 Student Tasks
* Task 1 (Legacy): Implement the run_legacy_agent Prompt Template to establish a baseline (langchain).

* Task 2 (Router): Implement the retrieve_node logic to route queries to "apple", "tesla", or "both".

* Task 3 (Grader): Implement the grade_documents_node to filter out irrelevant documents.

* Task 4 (Generator): Implement the generate_node to answer questions in English with Citations.

* Task 5 (Rewriter): Implement the rewrite_node to refine search queries when retrieval fails.

# 🚀 Execution Order

* Step1: `python build_rag.py`: Before running any agents, you must ingest the PDFs and convert them into vector embeddings. This allows you to experiment with different chunking strategies without re-running the evaluation logic every time.
* Step2: `python evaluator.py`: Once the database is ready, run the evaluator to benchmark your agent.
  

# Task-by-Task Acceptance Checklist

Use this checklist after each task is implemented.

## Task A (Legacy ReAct Prompt)
- [ ] Prompt includes `{tools}`, `{tool_names}`, `{input}`, `{agent_scratchpad}`.
- [ ] Prompt explicitly enforces `Question/Thought/Action/Action Input/Observation/Final Answer`.
- [ ] Final answer must be English.
- [ ] Agent must distinguish 2024/2023/2022 columns.
- [ ] If exact 2024 value is unavailable, response is exactly `I don't know`.

## Task B (Router)
- [ ] Router output is restricted to `apple|tesla|both|none`.
- [ ] Retriever selection is dynamic according to routing result.

## Task C (Relevance Grader)
- [ ] Grader output is strict binary `yes`/`no`.
- [ ] `no` triggers rewrite branch.

## Task D (Query Rewriter)
- [ ] Rewriter converts vague questions into financial/accounting terminology.
- [ ] Company names and year constraints are preserved.

## Task E (Final Generator)
- [ ] Generator uses only retrieved context.
- [ ] Final answer contains citation tags such as `[Source: Apple]`.
- [ ] Missing information leads to `I don't know` (no hallucination).

# Benchmark Commands (for report.pdf)

## 1) Legacy vs Graph
```bash
# LEGACY
python evaluator.py

# GRAPH (Windows PowerShell)
python -c "import evaluator; evaluator.TEST_MODE='GRAPH'; evaluator.run_evaluation()"
```

## 2) Embedding model comparison (at least 2 models)
```bash
# Example A
$env:LOCAL_EMBEDDING_MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
$env:DB_FOLDER='chroma_db_l12'
python build_rag.py

# Example B
$env:LOCAL_EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2'
$env:DB_FOLDER='chroma_db_l6'
python build_rag.py
```

## 3) Chunk size comparison
```bash
# Small
$env:CHUNK_SIZE='500';  $env:CHUNK_OVERLAP='100'; $env:DB_FOLDER='chroma_db_c500';  python build_rag.py
# Medium
$env:CHUNK_SIZE='1000'; $env:CHUNK_OVERLAP='200'; $env:DB_FOLDER='chroma_db_c1000'; python build_rag.py
# Large
$env:CHUNK_SIZE='2000'; $env:CHUNK_OVERLAP='300'; $env:DB_FOLDER='chroma_db_c2000'; python build_rag.py
```
