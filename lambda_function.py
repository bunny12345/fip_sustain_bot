import os
import io
import json
import tarfile
import tempfile
import traceback
import re
import boto3
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock

# Configs
S3_BUCKET = "faissindexingfip"
AWS_REGION = "eu-west-1"

# Bedrock model IDs
EMBED_MODEL_ID = "cohere.embed-v4:0"
LLM_MODEL_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "180"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Supported languages: code -> human-readable name used in the prompt.
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ro": "Romanian",
    "pt": "Portuguese",
    "it": "Italian",
    "de": "German",
}
DEFAULT_LANGUAGE = "en"


def normalize_language(value):
    """Map an incoming language value to a supported code, defaulting to English."""
    if not value:
        return DEFAULT_LANGUAGE
    code = str(value).strip().lower().replace("_", "-").split("-")[0]
    return code if code in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def index_key_for(language):
    """S3 key of the FAISS index archive for a language."""
    return f"faiss_index_{language}.tar.gz"


# In-memory caches, keyed per language so each index/answer set stays isolated.
# CACHE: {language: {question: response_body}}
CACHE = {}
# VECTORSTORE_CACHE: {language: FAISS vectorstore}
VECTORSTORE_CACHE = {}

# Reuse Bedrock runtime client across invocations in warm containers.
BEDROCK_RUNTIME_CLIENT = boto3.client("bedrock-runtime", region_name=AWS_REGION)

STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "to", "for", "in", "on", "of",
    "what", "where", "when", "how", "can", "i", "we", "you", "about", "this", "that"
}

RESOURCE_LINKS = {
    "sdg": "https://sdgs.un.org/goals",
    "npv": "https://www.investopedia.com/terms/n/npv.asp",
}

def get_cors_headers(event=None):
    """Return CORS headers for responses.

    If an Origin header is present in the `event`, return that origin
    when it is in the allowlist. Otherwise fall back to the default origin.
    """
    allowed_origins = {
        "https://learninggateway.eu",
        "https://sustaineu-platform.learninggateway.eu",
        "https://chatvistaai.com",
    }

    origin = None
    if event:
        headers = event.get("headers") or {}
        origin = headers.get("origin") or headers.get("Origin")

    allowed_origin = origin if origin in allowed_origins else "https://learninggateway.eu"

    return {
        "Access-Control-Allow-Origin": allowed_origin,
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "OPTIONS,POST"
    }

def download_and_extract_faiss(s3_key):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)

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

def load_vectorstore(language):
    # Return a cached vectorstore for this language if already loaded.
    if language in VECTORSTORE_CACHE:
        print(f"Using cached vectorstore for language '{language}'")
        return VECTORSTORE_CACHE[language]

    s3_key = index_key_for(language)
    print(f"Loading vectorstore for language '{language}' from s3://{S3_BUCKET}/{s3_key}")
    try:
        temp_dir = download_and_extract_faiss(s3_key)
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
        VECTORSTORE_CACHE[language] = vectorstore
        return vectorstore
    except Exception as faiss_error:
        print("FAISS loading failed:", str(faiss_error))
        traceback.print_exc()
        raise faiss_error

def build_prompt(docs, question, language_name="English"):
    template = """You are a concise and helpful assistant for an ESE course chatbot.
- Answer using ONLY the provided context.
- Write your entire answer in {language_name}, regardless of the language of the context.
- If the user asks about a specific module/unit, do not invent cross-module mappings.
- If context is partial, clearly state what is known and unknown.
- Keep the answer compact and scannable.
- Target about 60 to 100 words total.
- Prefer 1 short intro sentence plus up to 4 short bullet points.
- Avoid long explanations unless explicitly asked.

# Formatting rules (VERY IMPORTANT):
# - Write in plain text only. Do NOT use any Markdown syntax.
# - Do NOT use asterisks (*), underscores (_), backticks (`), or hash symbols (#) for emphasis or headings.
# - Do NOT wrap words in ** or __ to make them bold.
# - For a heading, just write the phrase on its own line followed by a colon (e.g. "Key differences:").
# - For lists, start each item on a new line beginning with "- " (a hyphen and a space) and nothing else.
# - Keep sentences readable and avoid any special formatting characters.

Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
    context_text = "\n\n".join(doc.page_content for doc in docs)
    prompt = PromptTemplate.from_template(template)
    return prompt.format(context=context_text, question=question, language_name=language_name)


def extract_focus_terms(question):
    """Extract high-signal terms so lexical intent is preserved (e.g., NPV)."""
    q = question.lower()
    tokens = re.findall(r"\b[a-z0-9\-]{2,}\b", q)
    acronyms = set(re.findall(r"\b[A-Z]{2,10}\b", question))

    focus = set()
    for t in tokens:
        if t in STOPWORDS:
            continue
        # Keep short technical terms like npv, irr, ghg, csrd.
        if len(t) <= 4 or len(t) >= 6:
            focus.add(t)

    for a in acronyms:
        focus.add(a.lower())

    module_match = re.search(r"\b(module|unit)\s*(\d{1,2})\b", q)
    module_num = module_match.group(2) if module_match else None
    return sorted(focus), module_num, sorted(a.lower() for a in acronyms)


def doc_matches_module(doc, module_num):
    if not module_num:
        return True
    hay = f"{doc.page_content}\n{json.dumps(doc.metadata, ensure_ascii=True)}".lower()
    return bool(re.search(rf"\b(module|unit)\s*{re.escape(module_num)}\b", hay))


def lexical_overlap_score(doc, focus_terms):
    if not focus_terms:
        return 0
    text = doc.page_content.lower()
    score = 0
    for t in focus_terms:
        if re.search(rf"\b{re.escape(t)}\b", text):
            score += 1
    return score


def fallback_answer(question, docs, module_num, missing_terms):
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})
    msg = [
        "I could not find enough high-confidence context in the indexed course material to answer this precisely.",
    ]
    if module_num:
        msg.append(f"Your question mentions Module/Unit {module_num}, but the retrieved context did not clearly match that module.")
    if missing_terms:
        msg.append("Key term(s) not found in retrieved context: " + ", ".join(sorted(missing_terms)) + ".")

    msg.extend([
        "Suggested next sources:",
        "- Check the ESE module pages/search for the exact term (e.g., NPV on page 347 if available).",
        "- Ask the course facilitator for module-specific clarification.",
        "- Use a trusted external explainer for background, then verify against ESE materials.",
    ])

    q = question.lower()
    if "sdg" in q:
        msg.append(f"- UN SDGs reference: {RESOURCE_LINKS['sdg']}")
    if "npv" in q:
        msg.append(f"- NPV reference: {RESOURCE_LINKS['npv']}")

    return {
        "answer": "\n".join(msg),
        "sources": sources,
        "fallback": True,
    }

def call_llm(prompt):
    model = ChatBedrock(
        model_id=LLM_MODEL_ID,
        client=BEDROCK_RUNTIME_CLIENT,
        provider="anthropic",
        model_kwargs={
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
        },
    )
    return model.invoke(prompt)


def answer_mentions_required_acronyms(answer_text, acronyms):
    """Prevent acronym drift (e.g., user asks NPV but answer talks about GWP)."""
    if not acronyms:
        return True
    hay = (answer_text or "").lower()
    return all(a in hay for a in acronyms)

def lambda_handler(event, context):
    cors_headers = get_cors_headers(event)
    
    try:
        print("Received event:", json.dumps(event))

        # Handle CORS preflight
        if event.get("httpMethod") == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": cors_headers,
                "body": json.dumps({"message": "CORS preflight success"})
            }

        question = event.get("question")
        language = event.get("language")
        if "body" in event and event["body"]:
            body = json.loads(event["body"])
            if not question:
                question = body.get("question")
            if not language:
                language = body.get("language")

        language = normalize_language(language)
        language_name = SUPPORTED_LANGUAGES[language]
        print(f"Resolved language: {language} ({language_name})")

        if not question:
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "Missing 'question'"})
            }

        # Check cache first (per language).
        lang_cache = CACHE.setdefault(language, {})
        if question in lang_cache:
            print("Returning cached answer")
            cached_response = lang_cache[question]
            return {
                "statusCode": 200,
                "headers": cors_headers,
                "body": json.dumps(cached_response)
            }

        print("About to load vectorstore...")
        try:
            vectorstore = load_vectorstore(language)
            print("Vectorstore loaded successfully")
        except Exception as vs_error:
            print("Vectorstore loading failed with error:", str(vs_error))
            print("Vectorstore error type:", type(vs_error).__name__)
            traceback.print_exc()
            raise vs_error

        print("About to search docs...")
        try:
            scored = vectorstore.similarity_search_with_score(question, k=6)
            focus_terms, module_num, acronyms = extract_focus_terms(question)

            # Filter for explicit module if requested.
            filtered = [(d, s) for (d, s) in scored if doc_matches_module(d, module_num)]
            candidates = filtered if filtered else scored

            # Re-rank using lexical overlap first, then semantic distance score.
            ranked = sorted(
                candidates,
                key=lambda x: (-lexical_overlap_score(x[0], focus_terms), x[1])
            )
            docs = [d for (d, _) in ranked[:3]]

            all_context = "\n\n".join(d.page_content.lower() for d in docs)
            # Only use short acronym-like terms as hard confidence gates to avoid false fallback.
            critical_terms = set(acronyms)
            missing_terms = {t for t in critical_terms if t not in all_context}

            print("Docs loaded:", len(docs))
            print("Doc sources:", [doc.metadata.get("source", "unknown") for doc in docs])
            print("Focus terms:", focus_terms)
            print("Module requested:", module_num)
            print("Acronyms:", acronyms)
            print("Missing focus terms:", sorted(missing_terms))
        except Exception as search_error:
            print("Doc search failed with error:", str(search_error))
            print("Search error type:", type(search_error).__name__)
            traceback.print_exc()
            raise search_error

        # Confidence gate: if key short terms are missing, avoid hallucinated answers.
        if missing_terms:
            response_body = fallback_answer(question, docs, module_num, missing_terms)
        else:
            prompt = build_prompt(docs, question, language_name)
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

            answer_text = getattr(llm_response, 'content', None)

            # Secondary safety check after generation.
            if not answer_mentions_required_acronyms(answer_text, acronyms):
                response_body = fallback_answer(question, docs, module_num, set(acronyms))
                response_body["acronym_guard_triggered"] = True
            else:
                response_body = {
                    "answer": answer_text,
                "sources": list({doc.metadata.get("source", "unknown") for doc in docs}),
                "fallback": False,
                }

        # Cache the answer (per language).
        lang_cache[question] = response_body

        return {
            "statusCode": 200,
            "headers": cors_headers,
            "body": json.dumps(response_body)
        }

    except Exception as e:
        error_message = str(e) or repr(e)
        print("Error type:", type(e).__name__)
        print("Error message:", error_message)
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": get_cors_headers(event),
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
