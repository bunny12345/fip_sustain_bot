import os
import io
import json
import tarfile
import tempfile
import traceback
import boto3
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock

# Configs
S3_BUCKET = "faissindexingirlcollege"
S3_KEY = "faiss_index.tar.gz"
AWS_REGION = "eu-west-1"

# Bedrock model IDs
EMBED_MODEL_ID = "cohere.embed-v4:0"
LLM_MODEL_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"

# Simple in-memory cache for question -> answer
CACHE = {}

def download_and_extract_faiss():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    response = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)

    temp_dir = tempfile.mkdtemp()
    tar_data = io.BytesIO(response["Body"].read())

    with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:
        members = tar.getmembers()
        in_subdir = any(m.name.startswith("faiss_index/") for m in members if m.isfile())

        for member in members:
            if member.isfile():
                if in_subdir and member.name.startswith("faiss_index/"):
                    member.name = member.name[len("faiss_index/"):]
                tar.extract(member, path=temp_dir)

    for file_name in ["index.faiss", "index.pkl"]:
        if not os.path.exists(os.path.join(temp_dir, file_name)):
            raise FileNotFoundError(f"Missing expected file: {file_name}")

    return temp_dir

def load_vectorstore():
    print("Loading vectorstore from S3...")
    try:
        temp_dir = download_and_extract_faiss()
        print("FAISS files extracted to:", temp_dir)
        print("Files in temp_dir:", os.listdir(temp_dir))
    except Exception as download_error:
        print("FAISS download/extraction failed:", str(download_error))
        traceback.print_exc()
        raise download_error

    print("Creating embeddings...")
    try:
        embeddings = BedrockEmbeddings(
            client=boto3.client("bedrock-runtime", region_name=AWS_REGION),
            model_id=EMBED_MODEL_ID,
            provider="cohere"
        )
        print("Embeddings created successfully")
    except Exception as embed_error:
        print("Embeddings creation failed:", str(embed_error))
        traceback.print_exc()
        raise embed_error

    print("Loading FAISS vectorstore...")
    try:
        vectorstore = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
        print("FAISS vectorstore loaded successfully")
        print("FAISS index dimension:", vectorstore.index.d)
        return vectorstore
    except Exception as faiss_error:
        print("FAISS loading failed:", str(faiss_error))
        traceback.print_exc()
        raise faiss_error

def build_prompt(docs, question):
    template = """You are a concise and helpful assistant.
- Answer briefly and clearly using no more than 2 short paragraphs.
- Avoid repetition or over-explaining.
- Limit your response to 300 tokens maximum."
Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
    context_text = "\n\n".join(doc.page_content for doc in docs)
    prompt = PromptTemplate.from_template(template)
    return prompt.format(context=context_text, question=question)

def call_llm(prompt):
    model = ChatBedrock(
        model_id=LLM_MODEL_ID,
        client=boto3.client("bedrock-runtime", region_name=AWS_REGION),
        provider="anthropic"
    )
    return model.invoke(prompt)

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        # Handle CORS preflight
        if event.get("httpMethod") == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({"message": "CORS preflight success"})
            }

        question = event.get("question")
        if not question and "body" in event:
            body = json.loads(event["body"])
            question = body.get("question")

        if not question:
            return {
                "statusCode": 400,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Missing 'question'"})
            }

        # Check cache first
        if question in CACHE:
            print("Returning cached answer")
            cached_response = CACHE[question]
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps(cached_response)
            }

        print("About to load vectorstore...")
        try:
            vectorstore = load_vectorstore()
            print("Vectorstore loaded successfully")
        except Exception as vs_error:
            print("Vectorstore loading failed with error:", str(vs_error))
            print("Vectorstore error type:", type(vs_error).__name__)
            traceback.print_exc()
            raise vs_error

        print("About to search docs...")
        try:
            if hasattr(vectorstore, 'embeddings'):
                query_vec = vectorstore.embeddings.embed_query(question)
                print("Query embedding length:", len(query_vec))
                print("FAISS index dimension:", vectorstore.index.d)
                if len(query_vec) != vectorstore.index.d:
                    raise ValueError(
                        f"Embedding dimension mismatch: query vector length {len(query_vec)} does not match FAISS index dimension {vectorstore.index.d}. "
                        "Rebuild your FAISS index with the same embedding model/version."
                    )
            docs = vectorstore.similarity_search(question, k=4)
            print("Docs loaded:", len(docs))
            print("Doc sources:", [doc.metadata.get("source", "unknown") for doc in docs])
        except Exception as search_error:
            print("Doc search failed with error:", str(search_error))
            print("Search error type:", type(search_error).__name__)
            traceback.print_exc()
            raise search_error

        prompt = build_prompt(docs, question)
        print("Prompt length:", len(prompt))
        print("Prompt preview:", prompt[:500])

        print("About to call LLM...")
        try:
            llm_response = call_llm(prompt)
            print("LLM call successful")
            print("LLM response type:", type(llm_response))
            print("LLM response repr:", repr(llm_response)[:1000])
        except Exception as llm_error:
            print("LLM call failed with error:", str(llm_error))
            print("LLM error type:", type(llm_error).__name__)
            traceback.print_exc()
            raise llm_error

        response_body = {
            "answer": getattr(llm_response, 'content', None),
            "sources": list({doc.metadata.get("source", "unknown") for doc in docs})
        }

        # Cache the answer
        CACHE[question] = response_body

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps(response_body)
        }

    except Exception as e:
        error_message = str(e) or repr(e)
        print("Error type:", type(e).__name__)
        print("Error message:", error_message)
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": error_message})
        }

if __name__ == "__main__":
    print("Test started")
    # Test event for local execution
    test_event = {
        "body": json.dumps({"question": "What is the capital of France?"}),
        "headers": {"Content-Type": "application/json"}
    }
    test_context = None  # Mock context if needed
    print("Before handler")
    try:
        result = lambda_handler(test_event, test_context)
        print("Test result:", result)
    except Exception as e:
        print("Test failed:", str(e))
        traceback.print_exc()
