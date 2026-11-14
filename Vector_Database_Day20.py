import chromadb
import os
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

# Create a persistent ChromaDB client
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create or access a collection
collection = client.get_or_create_collection(
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


api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def embedding(text):
    response = client.models.embed_content(
        model = 'gemini-embedding-2',
        contents=text
        )
    return response.embeddings[0].values


for item in all_chunks:
    item["embedding"] = embedding(item["text"])

# Store all chunks and embeddings in ChromaDB

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


user_query = input("Enter your query: ")

query_embedding = embedding(user_query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results = 3,
    include=['documents','metadatas','distances']
    )

print("\nTop matching results:")

for i in range(len(results["documents"][0])):

    print(f"\nResult {i + 1}")

    print("Source:", results["metadatas"][0][i]["source"])

    print("Chunk ID:", results["metadatas"][0][i]["chunk_id"])

    print("Distance:", results["distances"][0][i])

    print("Content:", results["documents"][0][i])


# Extract retrieved documents
retrieved_documents = results["documents"][0]

# Combine retrieved documents into context
context = "\n\n".join(retrieved_documents)

# Create prompt
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

# Generate answer using Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\nGenerated Answer:")
print(response.text)
