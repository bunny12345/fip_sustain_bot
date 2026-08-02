import io
import os
import sys
import tarfile
import tempfile

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import S3FileLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

AWS_REGION = os.getenv("AWS_REGION") or "eu-west-1"

# Bucket that holds the source course PDFs (one per language).
SOURCE_S3_BUCKET = (
    os.getenv("DOCUMENT_S3_BUCKET")
    or os.getenv("FAISS_SOURCE_S3_BUCKET")
    or "faissindexingfip"
)

# Bucket where the built FAISS indexes are uploaded (read by the Lambda).
INDEX_S3_BUCKET = os.getenv("INDEX_S3_BUCKET") or "faissindexingfip"

# Language configuration: code -> default source PDF key.
# Each key can be overridden with an env var DOCUMENT_S3_KEY_<CODE> (e.g. DOCUMENT_S3_KEY_RO).
LANGUAGES = {
    "en": "SUSTAIN EU_English content_final version.pdf",
    "ro": "SUSTAIN EU - ESE Course Romanian.pdf",
    "pt": "SUSTAIN EU - ESE course Portuguese.pdf",
    "it": "SUSTAIN EU - ESE Course Italian.pdf",
    "de": "SUSTAIN EU - ESE Course German.pdf",
}


def source_key_for(lang):
    """Resolve the source PDF key for a language, allowing env overrides."""
    return os.getenv(f"DOCUMENT_S3_KEY_{lang.upper()}") or LANGUAGES[lang]


def index_key_for(lang):
    """S3 key of the built index archive for a language."""
    return f"faiss_index_{lang}.tar.gz"


def s3_object_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as err:
        code = err.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def load_documents(bucket, key):
    if key.endswith(".tar.gz"):
        raise ValueError("Source key must point to the PDF file, not a .tar.gz")

    print(f"🔹 Loading documents from S3... bucket={bucket} key={key}")
    loader = S3FileLoader(bucket=bucket, key=key)
    return loader.load()


def create_embedder():
    print("🔹 Creating Cohere embedder...")
    return BedrockEmbeddings(
        model_id="cohere.embed-v4:0",
        region_name=AWS_REGION,
        provider="cohere",
    )


def build_vector_store(docs, embedder):
    print("🔹 Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("❌ No chunks created from documents. Check the source PDF contents.")

    print(f"🔹 Building FAISS vector store from {len(chunks)} chunks...")
    return FAISS.from_documents(chunks, embedder)


def archive_and_upload(s3, local_dir, bucket, key):
    """Tar+gzip a FAISS index directory and upload it to S3."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for file_name in ["index.faiss", "index.pkl"]:
            file_path = os.path.join(local_dir, file_name)
            tar.add(file_path, arcname=file_name)
    buffer.seek(0)

    print(f"🔹 Uploading index to s3://{bucket}/{key}")
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def build_language(lang, embedder, s3):
    source_key = source_key_for(lang)

    if not s3_object_exists(s3, SOURCE_S3_BUCKET, source_key):
        print(f"⏭️  Skipping '{lang}': source PDF not found (s3://{SOURCE_S3_BUCKET}/{source_key})")
        return False

    print(f"\n===== Building index for language: {lang} =====")
    docs = load_documents(SOURCE_S3_BUCKET, source_key)
    db = build_vector_store(docs, embedder)

    with tempfile.TemporaryDirectory() as tmp:
        db.save_local(tmp)
        archive_and_upload(s3, tmp, INDEX_S3_BUCKET, index_key_for(lang))

        # Keep the legacy default key in sync with English for backward compatibility.
        if lang == "en":
            archive_and_upload(s3, tmp, INDEX_S3_BUCKET, "faiss_index.tar.gz")

    print(f"✅ Index for '{lang}' uploaded to s3://{INDEX_S3_BUCKET}/{index_key_for(lang)}")
    return True


def main():
    # Allow limiting to specific languages via BUILD_LANGUAGES="en,ro" (defaults to all).
    requested = os.getenv("BUILD_LANGUAGES")
    langs = (
        [code.strip() for code in requested.split(",") if code.strip()]
        if requested
        else list(LANGUAGES.keys())
    )

    s3 = boto3.client("s3", region_name=AWS_REGION)
    embedder = create_embedder()

    built = []
    for lang in langs:
        if lang not in LANGUAGES:
            print(f"⚠️  Unknown language code '{lang}', skipping")
            continue
        if build_language(lang, embedder, s3):
            built.append(lang)

    if not built:
        print("❌ No indexes were built. Check that at least one source PDF exists in S3.")
        sys.exit(1)

    print(f"\n✅ Built indexes for: {', '.join(built)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - surface any build error to CI
        print(f"❌ Error: {exc}")
        sys.exit(1)