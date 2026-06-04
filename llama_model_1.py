import os
import io
import json
import tarfile
import tempfile
import boto3
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.prompts import PromptTemplate

# Configs
S3_BUCKET = "faissindexingirlcollege"
S3_KEY = "faiss_index.tar.gz"
AWS_REGION = "eu-west-1"

# Bedrock model IDs
EMBED_MODEL_ID = "cohere.embed-v4:0"
LLM_INFERENCE_PROFILE_ARN = "arn:aws:bedrock:eu-west-1:931886962745:inference-profile/eu.meta.llama3-2-1b-instruct-v1:0"

# Simple in-memory cache dictionary
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
    temp_dir = download_and_extract_faiss()
    embeddings = BedrockEmbeddings(
        client=boto3.client("bedrock-runtime", region_name=AWS_REGION),
        model_id=EMBED_MODEL_ID,
        provider="cohere"
    )
    return FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)

def build_prompt(docs, question):
    template = """You are a concise and helpful assistant.
- Answer briefly and clearly using no more than 2 short paragraphs.
- Avoid repetition or over-explaining.
Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
    context_text = "\n\n".join(doc.page_content for doc in docs)
    prompt = PromptTemplate.from_template(template)
    return prompt.format(context=context_text, question=question)

def call_llm(prompt):
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    payload = {"prompt": prompt}

    response = client.invoke_model(
        modelId=LLM_INFERENCE_PROFILE_ARN,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload).encode("utf-8")
    )
    output = response["body"].read().decode()

    try:
        parsed = json.loads(output)
        return parsed.get("generation", output)  # return generated text or raw output
    except json.JSONDecodeError:
        return output

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
                    "Access-Control-Allow-Methods": "OPTIONS,POST",
                },
                "body": json.dumps({"message": "CORS preflight success"}),
            }

        question = event.get("question")
        if not question and "body" in event:
            body = json.loads(event["body"])
            question = body.get("question")

        if not question:
            return {
                "statusCode": 400,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Missing 'question'"}),
            }

        # Check cache for repeated questions
        if question in CACHE:
            print("Returning cached answer")
            cached_response = CACHE[question]
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST",
                },
                "body": json.dumps(cached_response),
            }

        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(question, k=4)
        prompt = build_prompt(docs, question)
        llm_response = call_llm(prompt)

        response_body = {
            "answer": llm_response,
            "sources": list({doc.metadata.get("source", "unknown") for doc in docs}),
        }

        # Cache the response for future identical requests
        CACHE[question] = response_body

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
            },
            "body": json.dumps(response_body),
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
