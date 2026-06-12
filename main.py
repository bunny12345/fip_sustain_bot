import boto3
import os
import faiss
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import S3FileLoader
from langchain_aws import BedrockEmbeddings
def load_documents():
    bucket = os.getenv("FAISS_S3_BUCKET", "irlcolleges")
    key = os.getenv("FAISS_S3_KEY", "SUSTAIN-EU_Module 7_Carbon Accounting and Reporting_SRC_V2 (1).pdf")
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