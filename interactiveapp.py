import os
import time
import tempfile
from typing import Dict, List, Tuple

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)

# =========================================================
# CONFIG
# =========================================================
CEREBRAS_API_KEY = "ur_key"
GROQ_API_KEY = "ur_key"
MISTRAL_API_KEY = "ur_key"

CHUNK_SIZE = 300
TOP_K_DEFAULT = 3
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TIMEOUT = 120

AVAILABLE_MODELS = {
    "Cerebras": {
        "client_type": "openai_compatible",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key": CEREBRAS_API_KEY,
        "models": [
            "llama3.1-8b"
        ]
    },
    "Groq": {
        "client_type": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "models": [
            "llama-3.3-70b-versatile"
        ]
    },
    "Mistral": {
        "client_type": "mistral_http",
        "base_url": "https://api.mistral.ai/v1",
        "api_key": MISTRAL_API_KEY,
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "ministral-8b-latest"
        ]
    }
}

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}
.small-muted {
    color: #888;
    font-size: 0.9rem;
}
.answer-box {
    padding: 0.9rem 1rem;
    border-radius: 0.75rem;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CLIENTS
# =========================================================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def get_openai_compatible_client(base_url: str, api_key: str):
    return OpenAI(api_key=api_key, base_url=base_url)

embed_model = load_embed_model()

# =========================================================
# HELPERS
# =========================================================
def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, size: int) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

def embed_text(text: str):
    return embed_model.encode(text, convert_to_numpy=True)

def embed_texts(texts: List[str]):
    return embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

@st.cache_data(show_spinner=False)
def process_pdf_corpus(file_signatures: Tuple[Tuple[str, int], ...], file_paths: Tuple[str, ...], chunk_size: int):
    all_chunks = []

    for path, signature in zip(file_paths, file_signatures):
        display_name, _ = signature
        text = extract_pdf_text(path)
        chunks = chunk_text(text, chunk_size)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": display_name,
                "chunk_id": i
            })

    if not all_chunks:
        return [], None

    chunk_texts = [item["text"] for item in all_chunks]
    chunk_embeddings = embed_texts(chunk_texts)
    return all_chunks, chunk_embeddings

def retrieve_chunks(question: str, chunks: List[Dict], chunk_embeddings, top_k: int):
    q_emb = embed_text(question).reshape(1, -1)
    sims = cosine_similarity(q_emb, chunk_embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    return [
        {
            "text": chunks[i]["text"],
            "source": chunks[i]["source"],
            "chunk_id": chunks[i]["chunk_id"],
            "score": float(sims[i])
        }
        for i in top_idx
    ]

def call_openai_compatible_llm(client, model_name: str, prompt: str, temperature: float = 0.1, timeout: int = DEFAULT_TIMEOUT) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout
    )
    return response.choices[0].message.content.strip()

def call_mistral_llm(model_name: str, prompt: str, temperature: float = 0.1, timeout: int = DEFAULT_TIMEOUT, max_retries: int = 3) -> str:
    import requests

    url = f"{AVAILABLE_MODELS['Mistral']['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AVAILABLE_MODELS['Mistral']['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))

    raise last_error

def generate_rag_answer(provider_name: str, model_name: str, question: str, contexts: List[str], temperature: float = 0.1) -> str:
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a retrieval-augmented question answering assistant.

Answer the question using only the provided context.

Instructions:
- Read the context carefully.
- Identify the parts relevant to the question.
- Answer clearly and accurately.
- Keep the answer concise, but include necessary detail if it improves correctness.
- Do not use outside knowledge.
- If the answer is not present in the context, reply exactly with:
"The answer is not available in the provided context."

Context:
{context_text}

Question:
{question}

Answer:
"""

    try:
        provider_info = AVAILABLE_MODELS[provider_name]

        if provider_info["client_type"] == "openai_compatible":
            client = get_openai_compatible_client(
                provider_info["base_url"],
                provider_info["api_key"]
            )
            return call_openai_compatible_llm(
                client=client,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                timeout=DEFAULT_TIMEOUT
            )

        if provider_info["client_type"] == "mistral_http":
            return call_mistral_llm(
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                timeout=DEFAULT_TIMEOUT
            )

        return f"[ERROR with {provider_name}/{model_name}: Unsupported provider type]"

    except Exception as e:
        return f"[ERROR with {provider_name}/{model_name}: {e}]"

def build_file_signature(pdf_items: List[Dict]) -> Tuple[Tuple[str, int], ...]:
    signatures = []
    for item in pdf_items:
        size = 0
        try:
            size = os.path.getsize(item["path"])
        except Exception:
            pass
        signatures.append((item["name"], size))
    return tuple(signatures)

def build_model_choices() -> List[str]:
    choices = []
    for provider_name, provider_info in AVAILABLE_MODELS.items():
        for model_name in provider_info["models"]:
            choices.append(f"{provider_name} :: {model_name}")
    return choices

def parse_model_choice(choice: str) -> Tuple[str, str]:
    provider_name, model_name = choice.split(" :: ", 1)
    return provider_name, model_name

# =========================================================
# HEADER
# =========================================================
st.title("📄 PDF RAG Assistant")
st.caption("Ask questions over local or uploaded PDFs, and choose exactly which model should answer.")

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.header("Settings")

    model_choices = build_model_choices()

    primary_model_choice = st.selectbox(
        "Primary answering model",
        options=model_choices,
        index=0
    )

    compare_mode = st.checkbox("Compare with additional models", value=False)

    comparison_choices = []
    if compare_mode:
        comparison_choices = st.multiselect(
            "Additional models to compare",
            options=[m for m in model_choices if m != primary_model_choice],
            default=[]
        )

    top_k = st.slider("Top K retrieved chunks", min_value=1, max_value=8, value=TOP_K_DEFAULT)
    chunk_size = st.slider("Chunk size (words)", min_value=100, max_value=600, value=CHUNK_SIZE, step=50)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

    st.markdown("---")
    st.markdown("### Current usage strategy")
    st.markdown(
        '<div class="small-muted">Only the selected model is called unless comparison mode is enabled.</div>',
        unsafe_allow_html=True
    )

# =========================================================
# FILE INPUT
# =========================================================
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("PDF sources")

    local_pdf_files = [f for f in os.listdir() if f.lower().endswith(".pdf")]
    selected_local_files = st.multiselect(
        "Choose local PDF files",
        local_pdf_files
    )

    uploaded_files = st.file_uploader(
        "Or upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

with right_col:
    st.subheader("Question")
    question = st.text_area(
        "Ask a question about the PDFs",
        height=120,
        placeholder="Example: What are the three core components of a mechatronic system?"
    )

pdf_items = []

for filename in selected_local_files:
    pdf_items.append({
        "path": filename,
        "name": os.path.basename(filename)
    })

if uploaded_files:
    for uploaded_file in uploaded_files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded_file.read())
        tmp.close()
        pdf_items.append({
            "path": tmp.name,
            "name": uploaded_file.name
        })

# =========================================================
# RUN
# =========================================================
run_clicked = st.button("Get Answer", use_container_width=True)

if run_clicked:
    if not pdf_items:
        st.error("Please choose or upload at least one PDF.")
    elif not question.strip():
        st.error("Please type a question.")
    else:
        selected_models = [primary_model_choice] + comparison_choices
        selected_models = list(dict.fromkeys(selected_models))

        with st.spinner("Reading PDFs, retrieving chunks, and generating answer(s)..."):
            file_paths = tuple(item["path"] for item in pdf_items)
            file_signatures = build_file_signature(pdf_items)

            all_chunks, chunk_embeddings = process_pdf_corpus(
                file_signatures=file_signatures,
                file_paths=file_paths,
                chunk_size=chunk_size
            )

            if not all_chunks or chunk_embeddings is None:
                st.error("No text could be extracted from the selected PDFs.")
                st.stop()

            retrieved_items = retrieve_chunks(
                question=question,
                chunks=all_chunks,
                chunk_embeddings=chunk_embeddings,
                top_k=top_k
            )
            retrieved_chunks = [item["text"] for item in retrieved_items]

            answers = []
            for model_choice in selected_models:
                provider_name, model_name = parse_model_choice(model_choice)
                answer = generate_rag_answer(
                    provider_name=provider_name,
                    model_name=model_name,
                    question=question,
                    contexts=retrieved_chunks,
                    temperature=temperature
                )
                answers.append({
                    "provider": provider_name,
                    "model": model_name,
                    "answer": answer
                })

        st.subheader("Answer(s)")

        if len(answers) == 1:
            ans = answers[0]
            st.markdown(
                f"<div class='answer-box'><strong>{ans['provider']} — {ans['model']}</strong><br><br>{ans['answer']}</div>",
                unsafe_allow_html=True
            )
        else:
            cols = st.columns(min(len(answers), 3))
            for idx, ans in enumerate(answers):
                with cols[idx % len(cols)]:
                    st.markdown(
                        f"<div class='answer-box'><strong>{ans['provider']} — {ans['model']}</strong><br><br>{ans['answer']}</div>",
                        unsafe_allow_html=True
                    )

        with st.expander("Retrieved chunks", expanded=False):
            for i, item in enumerate(retrieved_items, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(f"**Source:** {item['source']}")
                st.write(f"**Chunk ID:** {item['chunk_id']}")
                st.write(f"**Similarity:** {item['score']:.4f}")
                st.write(item["text"])
                st.markdown("---")

        with st.expander("Run summary", expanded=False):
            st.write("**Selected PDFs:**")
            for item in pdf_items:
                st.write(f"- {item['name']}")

            st.write("**Models used:**")
            for ans in answers:
                st.write(f"- {ans['provider']} / {ans['model']}")

            st.write(f"**Top K:** {top_k}")
            st.write(f"**Chunk size:** {chunk_size}")
            st.write(f"**Temperature:** {temperature}")
