from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from datetime import date , datetime
import os
import json
import requests
import firebase_admin
from firebase_admin import credentials, messaging
from fastapi import UploadFile, File, Form
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import pdfplumber
import io
import time

load_dotenv()
firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
if firebase_creds_json:
    cred = credentials.Certificate(json.loads(firebase_creds_json))
    firebase_admin.initialize_app(cred)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
supabase_url = os.getenv("SUPABASE_URL")
supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
daily_request_limit = int(os.getenv("DAILY_REQUEST_LIMIT", "25"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

# --- Gemini call hardening: retry transient failures with exponential backoff ---

def _is_transient(exc: Exception) -> bool:
    """Retry on rate-limit / server / network blips (per-minute limits recover
    quickly). A truly exhausted daily quota will fail after the retries and be
    surfaced gracefully by the callers."""
    msg = str(exc).lower()
    return any(k in msg for k in (
        "429", "500", "502", "503", "504", "rate", "quota", "exhausted",
        "unavailable", "overloaded", "timeout", "deadline", "connection",
    ))


_gemini_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)


@_gemini_retry
def _embed(text: str):
    return client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )


@_gemini_retry
def _generate(prompt: str):
    return client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
    )


def get_embedding(text: str) -> list[float]:
    result = _embed(text)
    return result.embeddings[0].values

class ParseRequest(BaseModel):
    text: str
    user_id: str | None = None


def resolve_user_id(request: ParseRequest, x_user_id: str | None) -> str | None:
    return request.user_id or x_user_id


def get_usage_count(user_id: str, usage_date: str) -> int:
    if not supabase_url or not supabase_service_role_key:
        raise RuntimeError("Supabase credentials are not configured")
    response = requests.get(
        f"{supabase_url}/rest/v1/usage_log",
        params={
            "select": "request_count",
            "user_id": f"eq.{user_id}",
            "usage_date": f"eq.{usage_date}",
            "limit": 1,
        },
        headers={
            "apikey": supabase_service_role_key,
            "Authorization": f"Bearer {supabase_service_role_key}",
        },
        timeout=10,
    )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        return 0
    return int(rows[0].get("request_count", 0))


def save_usage_count(user_id: str, usage_date: str, request_count: int) -> None:
    if not supabase_url or not supabase_service_role_key:
        raise RuntimeError("Supabase credentials are not configured")
    response = requests.post(
        f"{supabase_url}/rest/v1/usage_log",
        json={
            "user_id": user_id,
            "usage_date": usage_date,
            "request_count": request_count,
        },
        params={"on_conflict": "user_id,usage_date"},
        headers={
            "apikey": supabase_service_role_key,
            "Authorization": f"Bearer {supabase_service_role_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        },
        timeout=10,
    )
    response.raise_for_status()


def enforce_rate_limit(user_id: str | None) -> None:
    """Per-user daily cap on Gemini-backed requests. Checks + increments in one
    place so every AI endpoint is protected (not just parse-input/assistant).
    Fails open on a usage-log outage so a transient DB blip never blocks users."""
    if not user_id:
        raise HTTPException(status_code=401, detail="user_id is required")
    today = date.today().isoformat()
    try:
        current = get_usage_count(user_id, today)
    except Exception as e:
        print(f"Usage check failed (allowing request): {e}")
        return
    if current >= daily_request_limit:
        raise HTTPException(
            status_code=429,
            detail=f"You've reached today's limit of {daily_request_limit} requests. Try again tomorrow.",
        )
    try:
        save_usage_count(user_id, today, current + 1)
    except Exception as e:
        print(f"Usage save failed (allowing request): {e}")


@app.post("/parse-input")
def parse_input(request: ParseRequest, x_user_id: str | None = Header(default=None)):
    today = date.today().isoformat()
    user_id = resolve_user_id(request, x_user_id)
    enforce_rate_limit(user_id)

    prompt = f"""Today's date is {today}.
Extract structured info from this input. Return ONLY valid JSON, no other text, no markdown formatting.
Input: "{request.text}"
Return JSON in this exact format:
{{
  "type": "task" or "reminder" or "note",
  "title": "a clean, short title",
  "due_date": "YYYY-MM-DD or null if no date mentioned"
}}
If a relative date is mentioned (e.g. "Friday", "tomorrow", "next week"), calculate the actual date based on today's date."""

    parsed = parse_llm_json(generate_text(prompt))
    if not parsed:
        parsed = {"type": "task", "title": request.text, "due_date": None}

    return parsed


@app.get("/")
def health_check():
    return {"status": "backend is running"}
class PushTestRequest(BaseModel):
    fcm_token: str
    title: str = "Test Reminder"
    body: str = "This is a test push notification"


@app.post("/send-test-push")
def send_test_push(request: PushTestRequest):
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=request.title,
                body=request.body,
            ),
            token=request.fcm_token,
        )
        response = messaging.send(message)
        return {"success": True, "message_id": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Push failed: {str(e)}")
    
@app.post("/check-reminders")
def check_reminders():
    if not supabase_url or not supabase_service_role_key:
        raise RuntimeError("Supabase credentials are not configured")

    now = datetime.utcnow().isoformat()

    # Get due, unnotified tasks
    response = requests.get(
        f"{supabase_url}/rest/v1/tasks",
        params={
            "select": "id,user_id,title,due_date",
            "notified": "eq.false",
            "due_date": f"lte.{now}",
            "status": "eq.pending",
        },
        headers={
            "apikey": supabase_service_role_key,
            "Authorization": f"Bearer {supabase_service_role_key}",
        },
        timeout=10,
    )
    response.raise_for_status()
    due_tasks = response.json()

    sent_count = 0
    for task in due_tasks:
        token_response = requests.get(
            f"{supabase_url}/rest/v1/device_tokens",
            params={
                "select": "fcm_token",
                "user_id": f"eq.{task['user_id']}",
            },
            headers={
                "apikey": supabase_service_role_key,
                "Authorization": f"Bearer {supabase_service_role_key}",
            },
            timeout=10,
        )
        token_response.raise_for_status()
        tokens = token_response.json()

        for t in tokens:
            try:
                message = messaging.Message(
                    notification=messaging.Notification(
                        title="Task Reminder",
                        body=task["title"],
                    ),
                    token=t["fcm_token"],
                )
                messaging.send(message)
                sent_count += 1
            except Exception as e:
                print(f"Push failed for token {t['fcm_token']}: {e}")

        # Mark as notified
        requests.patch(
            f"{supabase_url}/rest/v1/tasks",
            params={"id": f"eq.{task['id']}"},
            json={"notified": True},
            headers={
                "apikey": supabase_service_role_key,
                "Authorization": f"Bearer {supabase_service_role_key}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    return {"checked": len(due_tasks), "notifications_sent": sent_count}
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    user_id: str


def get_recent_tasks(user_id: str) -> list[dict]:
    if not supabase_url or not supabase_service_role_key:
        return []
    response = requests.get(
        f"{supabase_url}/rest/v1/tasks",
        params={
            "select": "title,due_date,status",
            "user_id": f"eq.{user_id}",
            "order": "created_at.desc",
            "limit": 10,
        },
        headers={
            "apikey": supabase_service_role_key,
            "Authorization": f"Bearer {supabase_service_role_key}",
        },
        timeout=10,
    )
    if response.status_code != 200:
        return []
    return response.json()


@app.post("/chat")
def chat(request: ChatRequest):
    enforce_rate_limit(request.user_id)

    tasks = get_recent_tasks(request.user_id)
    tasks_summary = "\n".join(
        f"- {t['title']} (due: {t.get('due_date') or 'no date'}, status: {t['status']})"
        for t in tasks
    ) or "No tasks currently."

    system_context = f"""You are a warm, supportive personal companion inside a student's task management app. The student may share how they're feeling, vent about stress, or talk about their day.

Be genuinely warm and present. Keep responses conversational and fairly short (2-4 sentences), like a caring friend, not a therapist giving a lecture. You can reference their tasks below if relevant to what they're saying, but don't force it in.

If they mention something like a low score or a setback, respond with real empathy first, before any advice. If someone seems to be going through something serious or heavy (not just everyday stress), gently encourage them to talk to someone they trust or a counselor, without being alarmist about it.

Their recent tasks:
{tasks_summary}
"""

    contents = system_context + "\n\nConversation so far:\n"
    for m in request.messages:
        contents += f"{m.role}: {m.content}\n"
    contents += "assistant:"

    try:
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=contents,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")

    return {"reply": response.text.strip()}
MAX_CHUNKS_PER_DOCUMENT = 300


def _set_document_status(document_id: str, status: str) -> None:
    try:
        requests.patch(
            f"{supabase_url}/rest/v1/documents",
            params={"id": f"eq.{document_id}"},
            json={"status": status},
            headers={
                "apikey": supabase_service_role_key,
                "Authorization": f"Bearer {supabase_service_role_key}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
    except Exception as e:
        print(f"Failed to set document {document_id} status={status}: {e}")


def process_document_chunks(document_id: str, user_id: str, full_text: str) -> None:
    """Background job: chunk -> embed -> store, then flip the document status to
    'ready' or 'failed'. Runs after the upload response is returned so a large
    PDF never blocks or times out the request. Marking 'failed' (instead of the
    old always-'ready') means a doc that produced zero chunks no longer looks
    usable when it isn't."""
    try:
        chunks = chunk_text(full_text)[:MAX_CHUNKS_PER_DOCUMENT]
        chunk_records = []
        for idx, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk)
                chunk_records.append({
                    "document_id": document_id,
                    "user_id": user_id,
                    "chunk_text": chunk,
                    "chunk_index": idx,
                    "embedding": embedding,
                })
            except Exception as e:
                print(f"Embedding failed for chunk {idx}: {e}")
            time.sleep(0.05)  # gentle pacing so a big doc doesn't burst the rate limit

        if not chunk_records:
            _set_document_status(document_id, "failed")
            return

        insert = requests.post(
            f"{supabase_url}/rest/v1/document_chunks",
            json=chunk_records,
            headers={
                "apikey": supabase_service_role_key,
                "Authorization": f"Bearer {supabase_service_role_key}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        if insert.status_code >= 400:
            print(f"Chunk insert failed: {insert.status_code} {insert.text}")
            _set_document_status(document_id, "failed")
            return

        _set_document_status(document_id, "ready")
    except Exception as e:
        print(f"Document processing failed for {document_id}: {e}")
        _set_document_status(document_id, "failed")


@app.post("/upload-document")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
):
    enforce_rate_limit(user_id)

    # Extract text from the PDF up front (fast); embedding happens in background.
    file_bytes = await file.read()
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    # Create the document record in 'processing' state.
    doc_response = requests.post(
        f"{supabase_url}/rest/v1/documents",
        json={"user_id": user_id, "title": file.filename, "status": "processing"},
        headers={
            "apikey": supabase_service_role_key,
            "Authorization": f"Bearer {supabase_service_role_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        timeout=10,
    )
    doc_response.raise_for_status()
    document = doc_response.json()[0]
    document_id = document["id"]

    # Heavy work runs after the response is sent; the app polls status.
    background_tasks.add_task(process_document_chunks, document_id, user_id, full_text)

    return {"document_id": document_id, "status": "processing"}


class AskDocumentRequest(BaseModel):
    question: str
    user_id: str
    document_id: str | None = None


@app.post("/ask-document")
def ask_document(request: AskDocumentRequest):
    enforce_rate_limit(request.user_id)

    try:
        matches = retrieve_chunks(request.question, request.user_id, request.document_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {str(e)}")

    if not matches:
        return {
            "answer": "I couldn't find anything relevant in your documents to answer that.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[Chunk {m['chunk_index']}] {m['chunk_text']}" for m in matches
    )

    answer = generate_text(
        f"""Answer the question using ONLY the context below, which was retrieved from the user's uploaded document(s). If the context doesn't contain enough information to answer, say so honestly instead of guessing.

Context:
{context}

Question: {request.question}

Answer:"""
    )

    return {
        "answer": answer,
        "sources": [
            {"document_id": m["document_id"], "chunk_index": m["chunk_index"]}
            for m in matches
        ],
    }


def supabase_headers(json_content: bool = False) -> dict:
    headers = {
        "apikey": supabase_service_role_key,
        "Authorization": f"Bearer {supabase_service_role_key}",
    }
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


# ---------------------------------------------------------------------------
# Hybrid retrieval: semantic (vector) + lexical (keyword) search, fused with
# Reciprocal Rank Fusion, then reranked by an LLM before answer generation.
# ---------------------------------------------------------------------------

def _vector_search(question: str, user_id: str, document_id: str | None, count: int = 20) -> list[dict]:
    query_embedding = get_embedding(question)
    resp = requests.post(
        f"{supabase_url}/rest/v1/rpc/match_document_chunks",
        json={
            "query_embedding": query_embedding,
            "match_user_id": user_id,
            "match_document_id": document_id,
            "match_count": count,
        },
        headers=supabase_headers(json_content=True),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _keyword_search(question: str, user_id: str, document_id: str | None, count: int = 20) -> list[dict]:
    # Degrades gracefully: if the keyword_search_chunks function isn't deployed
    # yet (migration 004 not run), fall back to vector-only retrieval.
    try:
        resp = requests.post(
            f"{supabase_url}/rest/v1/rpc/keyword_search_chunks",
            json={
                "query_text": question,
                "match_user_id": user_id,
                "match_document_id": document_id,
                "match_count": count,
            },
            headers=supabase_headers(json_content=True),
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        return resp.json()
    except Exception as e:
        print(f"Keyword search unavailable: {e}")
        return []


def _reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse multiple ranked lists into one. RRF score for an item is the sum of
    1/(k + rank) across every list it appears in; k dampens the weight of low
    ranks. Robust and parameter-light — no per-retriever score tuning needed."""
    scores: dict = {}
    meta: dict = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            meta.setdefault(cid, item)
    return sorted(meta.values(), key=lambda m: scores[m["id"]], reverse=True)


def _rerank(question: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """LLM cross-encoder-style rerank: score each candidate's relevance to the
    question directly, far more precise than embedding similarity. Uses Gemini
    (no extra dependency/key); a dedicated reranker (e.g. Cohere) can drop in
    here later if quality needs it."""
    if len(chunks) <= top_n:
        return chunks
    listing = "\n\n".join(f"[{i}] {c['chunk_text']}" for i, c in enumerate(chunks))
    prompt = f"""Rank the passages by how well they help answer the question.
Question: {question}

Passages:
{listing}

Return ONLY a JSON array of the {top_n} most relevant passage numbers, most relevant first. Example: [3, 0, 7, 1, 5]"""
    order = parse_llm_json(generate_text(prompt))
    if not isinstance(order, list):
        return chunks[:top_n]
    reranked = []
    for idx in order:
        if isinstance(idx, int) and 0 <= idx < len(chunks):
            reranked.append(chunks[idx])
        if len(reranked) >= top_n:
            break
    return reranked or chunks[:top_n]


def retrieve_chunks(question: str, user_id: str, document_id: str | None = None, top_n: int = 5) -> list[dict]:
    """Full hybrid retrieval pipeline. Returns the top_n most relevant chunks,
    each as {id, document_id, chunk_text, chunk_index}."""
    vector_results = _vector_search(question, user_id, document_id, count=20)
    keyword_results = _keyword_search(question, user_id, document_id, count=20)
    fused = _reciprocal_rank_fusion([vector_results, keyword_results])
    if not fused:
        return []
    return _rerank(question, fused[:12], top_n=top_n)


def list_user_documents(user_id: str) -> list[dict]:
    response = requests.get(
        f"{supabase_url}/rest/v1/documents",
        params={
            "select": "id,title,status",
            "user_id": f"eq.{user_id}",
            "status": "eq.ready",
            "order": "created_at.desc",
        },
        headers=supabase_headers(),
        timeout=10,
    )
    if response.status_code != 200:
        return []
    return response.json()


def fetch_document_chunks(document_id: str, limit: int = 30) -> list[str]:
    response = requests.get(
        f"{supabase_url}/rest/v1/document_chunks",
        params={
            "select": "chunk_text",
            "document_id": f"eq.{document_id}",
            "order": "chunk_index.asc",
            "limit": limit,
        },
        headers=supabase_headers(),
        timeout=10,
    )
    response.raise_for_status()
    return [row["chunk_text"] for row in response.json()]


def insert_task(user_id: str, title: str, raw_input: str, due_date: str | None) -> None:
    response = requests.post(
        f"{supabase_url}/rest/v1/tasks",
        json={
            "user_id": user_id,
            "title": title,
            "raw_input": raw_input,
            "type": "task",
            "due_date": due_date,
            "status": "pending",
        },
        headers=supabase_headers(json_content=True),
        timeout=10,
    )
    response.raise_for_status()


def generate_text(prompt: str) -> str:
    try:
        response = _generate(prompt)
    except Exception as e:
        # Retries exhausted (e.g. quota truly out / Gemini down). Surface a
        # friendly, retryable signal instead of a raw 500 / stack trace.
        print(f"LLM generation failed after retries: {e}")
        raise HTTPException(
            status_code=503,
            detail="The assistant is busy right now. Please try again in a moment.",
        )
    return response.text.strip()


def parse_llm_json(text: str) -> dict | None:
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@app.post("/assistant")
def assistant(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages is required")
    enforce_rate_limit(request.user_id)

    today = date.today().isoformat()
    latest = request.messages[-1].content
    documents = list_user_documents(request.user_id)
    docs_listing = "\n".join(
        f'- id: {d["id"]}, title: "{d["title"]}"' for d in documents
    ) or "(no documents uploaded)"

    intent_prompt = f"""Today's date is {today}. You are the intent router for a student's personal assistant app.
The user said: "{latest}"

The user's uploaded documents:
{docs_listing}

Classify the intent and return ONLY valid JSON, no other text, no markdown:
{{
  "intent": "add_task" or "ask_documents" or "summarize_document" or "chat",
  "title": "clean short task title, or null",
  "due_date": "YYYY-MM-DD or null (resolve relative dates from today)",
  "document_id": "id of the document the user refers to, or null",
  "question": "the question to ask over documents, or null"
}}

Rules:
- "add_task": user wants to add/remember a task, reminder, or deadline.
- "summarize_document": user asks to summarize/explain/give overview of one of their documents. Match the document by title, even loosely.
- "ask_documents": user asks a question whose answer would be in their uploaded documents (mentions a doc, or asks about study material content).
- "chat": everything else - feelings, greetings, general talk.
- If the user references a document that doesn't exist in the list, use intent "chat"."""

    parsed = parse_llm_json(generate_text(intent_prompt)) or {"intent": "chat"}
    intent = parsed.get("intent", "chat")

    if intent == "add_task" and parsed.get("title"):
        try:
            insert_task(request.user_id, parsed["title"], latest, parsed.get("due_date"))
        except Exception:
            raise HTTPException(status_code=502, detail="Failed to save task")
        due = parsed.get("due_date")
        reply = f"Added: {parsed['title']}" + (f", due {due}" if due else "")
        return {"reply": reply, "intent": intent}

    if intent == "summarize_document" and parsed.get("document_id"):
        chunks = fetch_document_chunks(parsed["document_id"])
        if not chunks:
            return {
                "reply": "I found that document but it has no readable content to summarize.",
                "intent": intent,
            }
        doc_text = "\n\n".join(chunks)
        summary = generate_text(
            f"""Summarize the following document for a student. Be clear and concise:
start with a 1-2 sentence overview, then the key points as short bullets.

Document:
{doc_text}

Summary:"""
        )
        return {"reply": summary, "intent": intent}

    if intent == "ask_documents":
        question = parsed.get("question") or latest
        try:
            matches = retrieve_chunks(question, request.user_id, parsed.get("document_id"))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Retrieval failed: {str(e)}")
        if not matches:
            return {
                "reply": "I couldn't find anything relevant in your documents to answer that.",
                "intent": intent,
            }
        context = "\n\n".join(
            f"[Chunk {m['chunk_index']}] {m['chunk_text']}" for m in matches
        )
        answer = generate_text(
            f"""Answer the question using ONLY the context below, which was retrieved from the user's uploaded document(s). If the context doesn't contain enough information to answer, say so honestly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""
        )
        return {"reply": answer, "intent": intent}

    # Default: companion chat (same behavior as /chat)
    tasks = get_recent_tasks(request.user_id)
    tasks_summary = "\n".join(
        f"- {t['title']} (due: {t.get('due_date') or 'no date'}, status: {t['status']})"
        for t in tasks
    ) or "No tasks currently."

    system_context = f"""You are a warm, supportive personal companion inside a student's task management app. The student may share how they're feeling, vent about stress, or talk about their day.

Be genuinely warm and present. Keep responses conversational and fairly short (2-4 sentences), like a caring friend, not a therapist giving a lecture. You can reference their tasks below if relevant to what they're saying, but don't force it in.

If they mention something like a low score or a setback, respond with real empathy first, before any advice. If someone seems to be going through something serious or heavy (not just everyday stress), gently encourage them to talk to someone they trust or a counselor, without being alarmist about it.

Their recent tasks:
{tasks_summary}
"""
    contents = system_context + "\n\nConversation so far:\n"
    for m in request.messages:
        contents += f"{m.role}: {m.content}\n"
    contents += "assistant:"

    return {"reply": generate_text(contents), "intent": "chat"}


class QuizRequest(BaseModel):
    user_id: str
    document_id: str
    num_questions: int = 5


@app.post("/generate-quiz")
def generate_quiz(request: QuizRequest):
    enforce_rate_limit(request.user_id)

    chunks = fetch_document_chunks(request.document_id, limit=40)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="This document has no readable content to quiz on.",
        )

    content = "\n\n".join(chunks)
    n = max(1, min(request.num_questions, 15))

    prompt = f"""You are an expert exam question writer for a college student.
Using ONLY the study material below, write {n} multiple-choice questions that test real understanding, not trivia.

Rules:
- Every question and its correct answer MUST be answerable from the material below. Do not use outside knowledge.
- Exactly 4 options each, exactly one correct.
- Make the wrong options plausible (common misconceptions), not obviously wrong.
- Give each question a short "topic" (2-4 words) naming the concept it covers.
- Add a one-sentence "explanation" of why the correct answer is right.

Return ONLY valid JSON, no markdown, in this exact shape:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "topic": "...",
      "explanation": "..."
    }}
  ]
}}

Study material:
{content}
"""

    parsed = parse_llm_json(generate_text(prompt))
    if not parsed or not isinstance(parsed.get("questions"), list):
        raise HTTPException(status_code=502, detail="Could not generate a quiz. Please try again.")

    questions = []
    for q in parsed["questions"]:
        if not isinstance(q, dict):
            continue
        opts = q.get("options")
        ci = q.get("correct_index")
        if not (isinstance(opts, list) and 2 <= len(opts) <= 6):
            continue
        if not isinstance(ci, int) or not (0 <= ci < len(opts)):
            continue
        questions.append({
            "question": str(q.get("question", "")).strip(),
            "options": [str(o) for o in opts],
            "correct_index": ci,
            "topic": str(q.get("topic", "General")).strip() or "General",
            "explanation": str(q.get("explanation", "")).strip(),
        })

    if not questions:
        raise HTTPException(status_code=502, detail="Could not generate valid questions. Please try again.")

    return {"questions": questions}