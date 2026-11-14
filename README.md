# Document QA using RAG

A simple document-grounded question-answering system built with **ChromaDB**, **Google Gemini embeddings**, and **Gemini 2.5 Flash**.

The project loads local text documents, splits them into overlapping chunks, converts those chunks into embeddings, stores them in a persistent ChromaDB collection, and retrieves the most relevant chunks to answer a user's question.

## How it works

1. **Load documents** – reads `.txt` files from the `documents/` folder.
2. **Chunk text** – splits each document into smaller, overlapping pieces.
3. **Create embeddings** – uses Google's `gemini-embedding-2` embedding model.
4. **Store vectors** – saves chunks, embeddings, and source metadata in ChromaDB.
5. **Retrieve context** – finds the top matching chunks for a question using vector similarity.
6. **Generate an answer** – sends the retrieved context and question to `gemini-2.5-flash`. The prompt asks the model to answer only from the retrieved context.

## Project structure

```text
Document-QA-using-RAG/
├── app.py
├── documents/
│   └── sample_document.txt
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The local ChromaDB data is created in `chroma_db/` when the app runs. It is excluded from Git.

## Requirements

- Python 3.10 or later
- A Google Gemini API key
- Internet access when creating embeddings and generating answers

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mathurlakshya/Document-QA-using-RAG.git
cd Document-QA-using-RAG
```

### 2. Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

Copy `.env.example` to a new file named `.env` and add your API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

The `.env` file is ignored by Git. **Never commit or share your real API key.**

### 5. Add your documents

Place your own plain-text `.txt` files inside the `documents/` folder. A small sample document is included so you can try the project immediately.

### 6. Run the project

```bash
python app.py
```

The app indexes the documents and stores their chunks in the local ChromaDB database. Then enter a question when prompted. Re-running the app upserts the same chunk IDs, so existing chunks are updated rather than duplicated.

## Example questions

Using the included sample document, try asking:

- What are the main stages of the water cycle?
- How does evaporation happen?
- What role does condensation play?

Answers depend on the content of the documents you provide.

## Tech stack

- **Python** – application logic
- **ChromaDB** – persistent vector database and similarity search
- **Google Gemini API** – text embeddings and answer generation
- **python-dotenv** – loading the API key from a local `.env` file

## Concepts explored

- Text chunking and overlap
- Vector embeddings
- Vector database storage
- Semantic similarity search
- Retrieval-Augmented Generation (RAG)
- Grounding LLM responses in retrieved context

## Current scope and limitations

- Supports plain-text `.txt` files only; PDF and Word parsing are not included.
- Chunks are created using a simple character-based strategy.
- The answer is based on the top retrieved chunks, so relevant information may be missed if retrieval is weak.
- This is an educational project, not a production-ready document search service.
- API calls require an internet connection and may incur usage charges under the provider's current terms.

## Privacy

Documents in `documents/` are read locally, but their text is sent to Google's API when embeddings are created. Retrieved text is also sent to Gemini when an answer is generated. Do not use confidential, personal, or sensitive documents unless you have reviewed the provider's data-handling terms and have permission to process them.

## Future improvements

- Add PDF and DOCX document support.
- Improve chunking with token-aware splitting.
- Add a Streamlit interface.
- Add source citations and retrieval evaluation.
- Support configurable retrieval settings and document management.

## Author

**Lakshya**  
GitHub: [@mathurlakshya](https://github.com/mathurlakshya)

---

Built as a hands-on project to understand vector databases, semantic search, and Retrieval-Augmented Generation.
