import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db" / "chroma_db"


def load_documents():
    """Load medical documents"""
    print("Loading medical documents...")

    file_path = DATA_DIR / "fake_medicine_data.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Medical data file not found: {file_path}"
        )

    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError("Medical dataset is empty")

    print(f"Loaded {len(documents)} document(s)")
    return documents


def split_documents(documents):
    """Split documents into chunks"""
    print("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    """Create vector database"""

    print("Creating vector database...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(DB_DIR),
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Vector database created successfully")
    print(f"Saved at: {DB_DIR}")

    return vectorstore


def main():

    print("Starting DocTalk RAG ingestion pipeline...\n")

    try:

        documents = load_documents()

        chunks = split_documents(documents)

        create_vector_store(chunks)

        print("\nRAG ingestion completed successfully")

    except Exception as error:

        print("\nERROR during ingestion pipeline")
        print(error)


if __name__ == "__main__":
    main()