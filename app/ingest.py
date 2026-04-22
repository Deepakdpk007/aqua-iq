import os
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

# ✅ Local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Persistent DB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_docs")


def load_pdfs(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))

            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            documents.append({
                "text": text,
                "source": file
            })
    return documents


def chunk_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)

    return chunks


def get_embedding(text):
    return model.encode(text).tolist()


def ingest():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data")

    docs = load_pdfs(DATA_PATH)

    id_counter = 0

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            embedding = get_embedding(chunk)

            collection.add(
                ids=[str(id_counter)],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc["source"]}]
            )

            id_counter += 1

    print("✅ Ingestion complete!")


if __name__ == "__main__":
    ingest()