import chromadb
import os
from google import genai

# Create a persistent ChromaDB client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create or access a collection
collection = chroma_client.get_or_create_collection(
    name="my_documents"
)

print("ChromaDB initialized successfully!")

DOCUMENTS_FOLDER = r"C:\Users\HP\Desktop\Python repos\Practice\documents"


def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            documents.append({
                "source": filename,
                "text": content
            })

    return documents


def chunk(text, chunk_size=1000, overlap=200):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be between 0 and chunk size")

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text
        })

        start += chunk_size - overlap
        chunk_id += 1

    return chunks


def keyword_search(query, all_chunks):
    query_words = query.lower().split()
    keyword_results = []

    for item in all_chunks:
        text = item["text"].lower()

        score = sum(
            1 for word in query_words
            if word in text
        )

        if score > 0:
            keyword_results.append({
                "source": item["source"],
                "text": item["text"],
                "keyword_score": score
            })

    keyword_results.sort(
        key=lambda item: item["keyword_score"],
        reverse=True
    )

    return keyword_results[:3]


# Load documents and split them into chunks
documents = load_documents(DOCUMENTS_FOLDER)

all_chunks = []

for document in documents:
    chunks = chunk(document["text"], 400, 10)

    for item in chunks:
        all_chunks.append({
            "source": document["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"]
        })


# Connect to Gemini
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing!")

client = genai.Client(api_key=api_key)


def embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )

    return response.embeddings[0].values


# Generate embeddings for all chunks
for item in all_chunks:
    item["embedding"] = embedding(item["text"])


# Store chunks, embeddings, and metadata in ChromaDB
collection.upsert(
    ids=[
        f"{item['source']}_{item['chunk_id']}"
        for item in all_chunks
    ],
    documents=[
        item["text"]
        for item in all_chunks
    ],
    embeddings=[
        item["embedding"]
        for item in all_chunks
    ],
    metadatas=[
        {
            "source": item["source"],
            "chunk_id": item["chunk_id"]
        }
        for item in all_chunks
    ]
)

print("Documents successfully stored in ChromaDB!")
print("Total stored chunks:", collection.count())


# Get the user's question
user_query = input("\nEnter your query: ")

query_embedding = embedding(user_query)


# Day 21: Configurable top-K
TOP_K = 5

# Day 21: Similarity distance threshold
# For ChromaDB's default L2 distance, smaller values indicate closer matches.
# This is an experimental value; tune it based on your own results.
MAX_DISTANCE = 1.0

# Day 21: Metadata filtering
# Set this to a filename, for example "python.txt", to search only that file.
# Keep it as None to search across all documents.
SOURCE_FILTER = None

query_options = {
    "query_embeddings": [query_embedding],
    "n_results": min(TOP_K, collection.count()),
    "include": ["documents", "metadatas", "distances"]
}

if SOURCE_FILTER:
    query_options["where"] = {"source": SOURCE_FILTER}

results = collection.query(**query_options)


# Keep only semantic results that pass the distance threshold
filtered_results = []

for i in range(len(results["documents"][0])):
    distance = results["distances"][0][i]

    if distance <= MAX_DISTANCE:
        filtered_results.append({
            "source": results["metadatas"][0][i]["source"],
            "chunk_id": results["metadatas"][0][i]["chunk_id"],
            "text": results["documents"][0][i],
            "distance": distance
        })


print("\nSemantic Search Results (after distance filtering):")

if filtered_results:
    for i, item in enumerate(filtered_results, start=1):
        print(f"\nResult {i}")
        print("Source:", item["source"])
        print("Chunk ID:", item["chunk_id"])
        print("Distance:", item["distance"])
        print("Content:", item["text"])
else:
    print("No results passed the distance threshold.")


# Day 21: Keyword search
keyword_results = keyword_search(user_query, all_chunks)

print("\nKeyword Search Results:")

if keyword_results:
    for i, item in enumerate(keyword_results, start=1):
        print(f"\nResult {i}")
        print("Source:", item["source"])
        print("Keyword Score:", item["keyword_score"])
        print("Content:", item["text"])
else:
    print("No keyword matches found.")


# Use only threshold-approved semantic results as context for the answer
retrieved_documents = [
    item["text"]
    for item in filtered_results
]

if not retrieved_documents:
    print("\nGenerated Answer:")
    print("I don't have enough relevant information in the retrieved documents to answer that.")
else:
    context = "\n\n".join(retrieved_documents)

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

If the answer is not available in the context,
say: "I don't have enough information to answer that."

Context:
{context}

Question:
{user_query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nGenerated Answer:")
    print(response.text)
