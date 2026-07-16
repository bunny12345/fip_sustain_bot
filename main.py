import os
import sys

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import S3FileLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


def load_documents():
    bucket = (
        os.getenv("DOCUMENT_S3_BUCKET")
        or os.getenv("FAISS_SOURCE_S3_BUCKET")
        or os.getenv("FAISS_S3_BUCKET")
        or "irlcolleges"
    )
    key = (
        os.getenv("DOCUMENT_S3_KEY")
        or os.getenv("FAISS_SOURCE_S3_KEY")
        or os.getenv("FAISS_S3_KEY")
        or "SUSTAIN EU_English content_final version.pdf"
    )

    if key.endswith(".tar.gz"):
        raise ValueError(
            "DOCUMENT_S3_KEY must point to the PDF file, not faiss_index.tar.gz"
        )

    print(f"🔹 Loading documents from S3... bucket={bucket} key={key}")
    loader = S3FileLoader(bucket=bucket, key=key)
    return loader.load()

def create_embedder():
    print("🔹 Creating Cohere embedder...")
    return BedrockEmbeddings(
        model_id="cohere.embed-v4:0",
        region_name="eu-west-1",
        provider="cohere"
    )

def build_vector_store(docs, embedder):
    print("🔹 Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("❌ No chunks created from documents. Check your S3 bucket path and contents.")

    print("🔹 Building FAISS vector store...")
    return FAISS.from_documents(chunks, embedder)

if __name__ == "__main__":
    try:
        docs = load_documents()
        embedder = create_embedder()
        db = build_vector_store(docs, embedder)
        db.save_local("faiss_index")
        print("✅ FAISS vector store saved locally to `faiss_index/`")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)