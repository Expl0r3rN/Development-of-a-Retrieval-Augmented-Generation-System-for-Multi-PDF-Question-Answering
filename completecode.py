
import json
import time
import requests
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer 
from openai import OpenAI  
PDF_FILES = [
    "file1.pdf",
    "file2.pdf",
    "file3.pdf"
] 

PINECONE_API_KEY = "ur_key"
CEREBRAS_API_KEY = "ur_key"
GROQ_API_KEY = "ur_key"
MISTRAL_API_KEY = "ur_key" 

INDEX_NAME = "optimus-rag"
CHUNK_SIZE = 300
TOP_K = 3

REFERENCE_FILE = "reference_qa.json" 
RAG_FILE = "rag_answers.json"
SCORES_FILE = "evaluation_scores.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMENSION = 384
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

CEREBRAS_MODELS = [
    "llama3.1-8b",
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
]

MISTRAL_MODELS = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "ministral-8b-latest",
]
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME in pc.list_indexes().names():
    pc.delete_index(INDEX_NAME)

pc.create_index(
    name=INDEX_NAME,
    dimension=EMBED_DIMENSION,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

time.sleep(5)
index = pc.Index(INDEX_NAME)

print("Pinecone ready.") 

cerebras_client = OpenAI(
    api_key=CEREBRAS_API_KEY,
    base_url="https://api.cerebras.ai/v1"
)

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

print("Clients ready.")

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("Embedding model loaded.")

def extract_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def chunk_text(text, size):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

def embed_text(text):
    return embed_model.encode(text, convert_to_numpy=True).tolist()

def embed_texts(texts, batch_size=32):
    return embed_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    ).tolist()


import requests
import time

def call_openai_compatible_llm(client, model_name, prompt, temperature=0.2, timeout=120):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout
    )
    return response.choices[0].message.content.strip()

def call_mistral_llm(model_name, prompt, temperature=0.2, timeout=120, max_retries=3):
    url = f"{MISTRAL_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
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

        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"Mistral request failed (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))

    raise last_error

def list_mistral_models():
    url = f"{MISTRAL_BASE_URL}/models"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and "data" in data:
            return [m.get("id") for m in data["data"] if "id" in m]
        elif isinstance(data, list):
            return [m.get("id") for m in data if "id" in m]
        else:
            return []
    except Exception as e:
        print(f"⚠️ Could not list Mistral models: {e}")
        return []

def test_openai_compatible_model_access(client, model_name, label):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply only with OK"}],
            temperature=0,
            timeout=30
        )
        print(f"✅ {label}: {model_name} works -> {response.choices[0].message.content.strip()}")
        return True
    except Exception as e:
        print(f"❌ {label}: {model_name} failed -> {e}")
        return False

def test_mistral_model_access(model_name):
    try:
        result = call_mistral_llm(model_name, "Reply only with OK", temperature=0, timeout=30)
        print(f"✅ Mistral: {model_name} works -> {result}")
        return True
    except Exception as e:
        print(f"❌ Mistral: {model_name} failed -> {e}")
        return False

  import re

def retrieve_chunks(question):
    q_emb = embed_text(question)
    results = index.query(
        vector=q_emb,
        top_k=TOP_K,
        include_metadata=True
    )
    return [match["metadata"]["text"] for match in results["matches"]]

def generate_rag_answer(provider, client_or_none, model_name, question, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a retrieval-augmented question answering assistant.

Answer the question using only the provided context.

Instructions:
- Read the context carefully.
- Identify the information relevant to the question.
- Answer clearly and accurately using only the context.
- You may include a little extra detail if it helps make the answer more complete.
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
        if provider == "mistral":
            return call_mistral_llm(model_name, prompt, temperature=0.1, timeout=120)
        else:
            return call_openai_compatible_llm(client_or_none, model_name, prompt, temperature=0.1, timeout=120)
    except Exception as e:
        return f"[ERROR with {provider}/{model_name}: {e}]"

def judge_answer(provider, client_or_none, model_name, question, reference, prediction):
    if prediction is None:
        return 0.0

    prediction_text = str(prediction).strip()

    if prediction_text.startswith("[ERROR"):
        return 0.0

    judge_prompt = f"""
You are a fair evaluator for a RAG system.

Question:
{question}

Reference Answer:
{reference}

Model Answer:
{prediction_text}

Scoring rules:
- Score the model answer from 0 to 5.
- Focus on factual correctness, relevance, completeness, and semantic similarity to the reference.
- Treat paraphrases and equivalent wording as correct.
- Do not punish the model for being slightly more detailed than the reference answer.
- If the model answer contains the correct answer plus extra correct supporting detail, score it highly.
- Only reduce the score if the extra detail is clearly incorrect, unsupported, or misleading.
- Ignore formatting differences such as bullets, bold text, numbering, or wording like "based on the context".
- If the model answer is shorter than the reference but still fully correct, score it highly.
- If the answer says the information is unavailable, give a high score only if the reference truly does not answer the question.
- If the model answer is an error message, irrelevant, or unsupported, give 0.

Use this scale:
5 = Correct and complete; may include extra correct detail
4 = Mostly correct; minor omission or minor unnecessary detail
3 = Partially correct
2 = Slightly correct but mostly wrong
1 = Incorrect
0 = Completely wrong, irrelevant, unsupported, or an error

Return only one number: 0, 1, 2, 3, 4, or 5.
"""

    try:
        if provider == "mistral":
            result = call_mistral_llm(model_name, judge_prompt, temperature=0.0, timeout=120)
        else:
            result = call_openai_compatible_llm(client_or_none, model_name, judge_prompt, temperature=0.0, timeout=120)

        raw_result = str(result).strip()
        match = re.search(r"\b([0-5])(?:\.0)?\b", raw_result)

        if match:
            return float(match.group(1))

        print(f"Could not parse judge output for {provider}/{model_name}: {raw_result}")
        return None

    except Exception as e:
        print(f"Judge failed for {provider}/{model_name}: {e}")
        return None

  print("Testing model access...")

active_models = []

for model_name in CEREBRAS_MODELS:
    if test_openai_compatible_model_access(cerebras_client, model_name, "Cerebras"):
        active_models.append({
            "provider": "cerebras",
            "client": cerebras_client,
            "model": model_name
        })

for model_name in GROQ_MODELS:
    if test_openai_compatible_model_access(groq_client, model_name, "Groq"):
        active_models.append({
            "provider": "groq",
            "client": groq_client,
            "model": model_name
        })

available_mistral_models = list_mistral_models()
if available_mistral_models:
    print("Mistral models visible to your key:")
    for m in available_mistral_models:
        print(" -", m)

for model_name in MISTRAL_MODELS:
    if test_mistral_model_access(model_name):
        active_models.append({
            "provider": "mistral",
            "client": None,
            "model": model_name
        })

if not active_models:
    raise RuntimeError("No working models were found.")

print("\nActive models:")
for m in active_models:
    print(f" - {m['provider']} / {m['model']}")
  print("Extracting and chunking PDFs...")
all_chunks = []

for pdf_file in PDF_FILES:
    print(f"Extracting {pdf_file}...")
    document_text = extract_pdf_text(pdf_file)
    chunks = chunk_text(document_text, CHUNK_SIZE)

    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "text": chunk,
            "source": pdf_file,
            "chunk_id": i
        })

print(f"Total chunks: {len(all_chunks)}")

print("Uploading to Pinecone...")
texts = [item["text"] for item in all_chunks]
chunk_embeddings = embed_texts(texts)

vectors = []
for i, (item, emb) in enumerate(zip(all_chunks, chunk_embeddings)):
    vectors.append({
        "id": str(i),
        "values": emb,
        "metadata": {
            "text": item["text"],
            "source": item["source"],
            "chunk_id": item["chunk_id"]
        }
    })

for i in range(0, len(vectors), 100):
    index.upsert(vectors=vectors[i:i + 100])

print("Upload complete.")

qa_pairs = [
    {"question": "What is a mechatronic system?", "answer": "A mechatronic system is the synergistic integration of mechanical design, electronics, sensing, and computer control to create functional and intelligent systems."},
    {"question": "What are the three core components of a mechatronic system?", "answer": "Actuators, sensors, and digital control devices."},
    {"question": "Why are humanoid robots considered mechatronic systems?", "answer": "Because they tightly integrate actuators, sensors, and digital control devices under synchronized communication and safety control."},
    {"question": "What motors are used in humanoid robot actuators?", "answer": "Permanent magnet synchronous motors or brushless DC motors."},
    {"question": "What is field oriented control?", "answer": "A control method used by servo drives for high-bandwidth torque regulation."},
    {"question": "Why are high reduction transmissions used?", "answer": "To convert motor speed to joint torque and achieve high joint torque at low speed."},
    {"question": "What is the role of electromechanical brakes?", "answer": "To provide safe holding and secure posture during power loss."},
    {"question": "Which sensors provide joint position feedback?", "answer": "Encoders or resolvers."},
    {"question": "What does an IMU measure?", "answer": "Angular rates and linear accelerations."},
    {"question": "Why are force sensors important?", "answer": "They measure interaction forces and enable safe contact and impedance or admittance control."},
    {"question": "What role does vision play in robots?", "answer": "Vision supports object detection, manipulation, and navigation."},
    {"question": "What do servo drives regulate?", "answer": "Torque, speed, and position."},
    {"question": "What safety feature disables torque?", "answer": "Safe Torque Off (STO)."},
    {"question": "Which protocol synchronizes robot components?", "answer": "EtherCAT."},
    {"question": "Which tasks show mechatronic integration?", "answer": "Walking, lifting, pushing, pulling, and manipulation."},
    {"question": "Why is torque density important?", "answer": "It enables high joint torque and precise low-speed control for compact actuators."},
    {"question": "What is whole body control?", "answer": "Coordinated control of all robot joints through a central real-time controller."},
    {"question": "Which sensors monitor system health?", "answer": "Current, voltage, temperature, and battery sensors."},
    {"question": "How does Tesla Optimus demonstrate mechatronics?", "answer": "Through the integration of actuators, sensors, and digital control devices in a coordinated humanoid system."},
    {"question": "How do humanoid actuators differ from EV motors?", "answer": "Humanoid actuators prioritize torque density and precision at low speed, while EV motors emphasize power at high speed."},
    {"question": "What is the advantage of a high torque motor?", "answer": "It enables high joint torque at low speed for precise and contact-rich tasks."},
    {"question": "Why use brushless DC motors in robots?", "answer": "Because they are used in electric rotary actuators with field-oriented control for precise torque regulation."},
    {"question": "What is a resolver used for?", "answer": "To measure position and speed feedback of a rotating shaft."},
    {"question": "How does an encoder work?", "answer": "It provides position and speed feedback by converting mechanical rotation into electrical signals."},
    {"question": "What is a servo drive?", "answer": "A digital control device that runs FOC, reads encoders and currents, and regulates torque, speed, and position."},
    {"question": "Why is Safe Torque Off important?", "answer": "It prevents unintended movement and supports safety by disabling torque."},
    {"question": "What is EtherCAT?", "answer": "A deterministic fieldbus for synchronized low-latency communication among sensors, drives, and controllers."},
    {"question": "What is whole body coordination?", "answer": "The simultaneous coordination of all robot joints for stable locomotion, interaction, and complex tasks."},
    {"question": "What types of sensors monitor voltage and current?", "answer": "Voltage and current sensors integrated in the system."},
    {"question": "Why is vision important for humanoid robots?", "answer": "It enables object detection, grasping, manipulation, and navigation."}
]

with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

print("Reference file saved.")
JUDGE_PROVIDER = "mistral"
JUDGE_MODEL = "mistral-medium-latest"
JUDGE_CLIENT = None

rag_results = []
scores = []

print("Running RAG with all active models...")

for q_idx, item in enumerate(qa_pairs, start=1):
    question = item["question"]
    reference = item["answer"]
    contexts = retrieve_chunks(question)

    print(f"\n[{q_idx}/{len(qa_pairs)}] {question}")

    result_row = {
        "question": question,
        "contexts": contexts,
        "reference": reference,
        "models": []
    }

    score_row = {
        "question": question,
        "reference": reference,
        "model_scores": []
    }

    for model_info in active_models:
        provider = model_info["provider"]
        client = model_info["client"]
        model_name = model_info["model"]

        print(f"  -> Testing {provider} / {model_name}")

        answer = generate_rag_answer(provider, client, model_name, question, contexts)

        # If answer generation itself failed, keep the error and score it as 0
        if isinstance(answer, str) and answer.startswith("[ERROR"):
            score = 0.0
            print(f"     answer failed")
        else:
            try:
                score = judge_answer(
                    JUDGE_PROVIDER,
                    JUDGE_CLIENT,
                    JUDGE_MODEL,
                    question,
                    reference,
                    answer
                )

                # If judge failed or returned None, mark as missing instead of fake zero
                if score is None:
                    print(f"     judge failed -> score=None")

            except Exception as e:
                print(f"     judge crashed -> {e}")
                score = None

        result_row["models"].append({
            "provider": provider,
            "model": model_name,
            "answer": answer
        })

        score_row["model_scores"].append({
            "provider": provider,
            "model": model_name,
            "answer": answer,
            "score": score
        })

    rag_results.append(result_row)
    scores.append(score_row)

print("\nRAG and scoring done.")

average_summary = []

for model_info in active_models:
    provider = model_info["provider"]
    model_name = model_info["model"]

    model_scores = []
    for row in scores:
        for ms in row["model_scores"]:
            if ms["provider"] == provider and ms["model"] == model_name:
                if ms["score"] is not None:
                    model_scores.append(ms["score"])

    avg_score = sum(model_scores) / len(model_scores) if model_scores else None

    average_summary.append({
        "provider": provider,
        "model": model_name,
        "average_score": avg_score
    })

average_summary = sorted(
    average_summary,
    key=lambda x: x["average_score"] if x["average_score"] is not None else -1,
    reverse=True
)
with open(RAG_FILE, "w", encoding="utf-8") as f:
    json.dump(rag_results, f, indent=2, ensure_ascii=False)

with open(SCORES_FILE, "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=2, ensure_ascii=False)

print("✅ Done!")
print(f"Saved: {RAG_FILE}")
print(f"Saved: {SCORES_FILE}")

print("Average scores:")
for item in average_summary:
    print(f"{item['provider']} / {item['model']} -> {item['average_score']:.3f}")
