import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "my_documents"

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 10
TOP_K = 3


def load_documents(folder: Path) -> list[dict]:
    """Load non-empty UTF-8 .txt files from the documents folder."""
    if not folder.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {folder}. Create it and add .txt files."
        )

    documents = []
    for file_path in sorted(folder.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            documents.append({"source": file_path.name, "text": content})
    return documents


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into character-based chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1.")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def create_embedding(gemini_client: genai.Client, text: str) -> list[float]:
    """Create a vector embedding with the Gemini embedding model."""
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    if not response.embeddings:
        raise RuntimeError("The embedding API returned no embedding.")
    return response.embeddings[0].values


def index_documents(
    documents: list[dict],
    collection,
    gemini_client: genai.Client,
) -> int:
    """Chunk, embed, and upsert documents into ChromaDB."""
    ids, texts, embeddings, metadatas = [], [], [], []

    for document in documents:
        for chunk_id, chunk in enumerate(chunk_text(document["text"])):
            ids.append(f'{document["source"]}_{chunk_id}')
            texts.append(chunk)
            embeddings.append(create_embedding(gemini_client, chunk))
            metadatas.append(
                {"source": document["source"], "chunk_id": chunk_id}
            )

    if ids:
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    return len(ids)


def retrieve_context(
    query: str,
    collection,
    gemini_client: genai.Client,
    top_k: int = TOP_K,
) -> tuple[str, list[dict]]:
    """Find the most relevant document chunks for a question."""
    query_vector = create_embedding(gemini_client, query)
    result_count = min(top_k, collection.count())
    if result_count == 0:
        return "", []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": text,
            "source": metadata.get("source", "Unknown"),
            "chunk_id": metadata.get("chunk_id", -1),
            "distance": distance,
        })

    context = "\n\n".join(
        f'Source: {item["source"]}\n{item["text"]}' for item in retrieved
    )
    return context, retrieved


def generate_answer(
    question: str,
    context: str,
    gemini_client: genai.Client,
) -> str:
    """Generate an answer using only the retrieved context."""
    prompt = f"""
You are a helpful assistant answering questions about the user's documents.
Use ONLY the context below to answer the question.
If the context does not contain the answer, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}
"""
    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text or "The model did not return a text response."


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. Add it to your .env file; "
            "see .env.example."
        )

    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        raise ValueError(
            f"No non-empty .txt files found in {DOCUMENTS_DIR}. "
            "Add at least one text file and run the app again."
        )

    # Keep the ChromaDB and Gemini clients separate.
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    gemini_client = genai.Client(api_key=api_key)

    print("Indexing documents...")
    indexed_count = index_documents(documents, collection, gemini_client)
    print(f"Indexed {indexed_count} chunks from {len(documents)} document(s).")
    print(f"Total chunks currently stored: {collection.count()}")

    question = input("\nEnter your question (or type 'exit' to quit): ").strip()
    if not question or question.lower() in {"exit", "quit"}:
        print("No question submitted. Exiting.")
        return

    context, retrieved = retrieve_context(question, collection, gemini_client)
    if not retrieved:
        print("No document chunks are available to search.")
        return

    print("\nRetrieved chunks:")
    for index, item in enumerate(retrieved, start=1):
        print(
            f'\n{index}. Source: {item["source"]} | '
            f'Chunk: {item["chunk_id"]} | Distance: {item["distance"]:.4f}'
        )
        print(item["text"])

    print("\nGenerated answer:")
    print(generate_answer(question, context, gemini_client))


if __name__ == "__main__":
    main()
