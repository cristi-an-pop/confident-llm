import json
import gradio as gr
import torch
import faiss
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Determine the script’s directory, then build absolute paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "finetuned_biogpt"))
INDEX_PATH  = os.path.abspath(os.path.join(BASE_DIR, "..", "faiss_index.index"))
META_PATH   = os.path.abspath(os.path.join(BASE_DIR, "..", "faiss_index_meta.json"))

# --- 1) Load your fine-tuned BioGPT model + tokenizer (local only) ---
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# --- 2) Load your FAISS index + metadata ---
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# --- 3) Load your Embedder ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- 4) Bring in your pipeline functions (or define inline) ---
def faiss_retrieve(query, index, metadata, embedder, top_k=3):
    q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_vec, top_k)
    return [metadata[i] for i in I[0] if i < len(metadata)]

def rerank_chunks(chunks, query):
    return sorted(chunks, key=lambda x: len(x["answer"]), reverse=True)

def rag_engine(query, chunks, model, tokenizer, max_length=256):
    if not chunks:
        return "No relevant context found."
    top = chunks[0]
    prompt = (
        "You are an expert in oral health.  Use ONLY the patient info below and do NOT repeat it.\n\n"
        f"Patient excerpt:\n"
        f"  • Q: {top['question']}\n"
        f"  • A: {top['answer']}\n\n"
        f"User question: {query}\n"
        "Answer concisely:"
    )
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        inputs,
        max_length=max_length,
        temperature=0.5,
        top_p=0.8,
        no_repeat_ngram_size=4,
        num_beams=2,
        early_stopping=True
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.replace(prompt, "").strip()

def toxicity_and_hallucination_filter(resp):
    blocked = ["badword1","badword2"]
    low = resp.lower()
    return "Content removed due to policy." if any(b in low for b in blocked) else resp

def generate_response(query):
    retrieved = faiss_retrieve(query, index, metadata, embedder, top_k=3)
    reranked  = rerank_chunks(retrieved, query)
    raw       = rag_engine(query, reranked, model, tokenizer)
    return toxicity_and_hallucination_filter(raw)

# --- 5) Override get_answer to call your real pipeline ---
def get_answer(user_message):
    return generate_response(user_message)

def update_history(user_message, history_json):
    history = json.loads(history_json) if history_json else []
    answer = get_answer(user_message)
    history.append({"question": user_message, "answer": answer, "feedback": None})
    return "", json.dumps(history), render_history(history)

def handle_feedback(feedback_data, history_json):
    if not feedback_data:
        return history_json, render_history(json.loads(history_json))
    idx_str, feedback = feedback_data.split(',')
    idx = int(idx_str)
    history = json.loads(history_json) if history_json else []
    if 0 <= idx < len(history):
        history[idx]["feedback"] = feedback
        print(f"Feedback: {feedback} for Q: '{history[idx]['question']}' -> A: '{history[idx]['answer']}'")
    return json.dumps(history), render_history(history)

def render_history(history):
    html_content = ""
    for i, turn in enumerate(history):
        question = turn["question"]
        answer = turn["answer"]
        fb = turn.get("feedback")
        feedback_display = f"Feedback: {fb}" if fb else ""
        html_content += f"""
        <div class="bubble-container">
            <div class="bubble-user">{question}</div>
            <div class="bubble-answer">
                {answer}
                <div class="feedback-buttons">
                    <button onclick="sendFeedback({i}, 'like')" style="border:none; background:none; cursor:pointer;">👍</button>
                    <button onclick="sendFeedback({i}, 'dislike')" style="border:none; background:none; cursor:pointer;">👎</button>
                </div>
                <div class="feedback-text">{feedback_display}</div>
            </div>
        </div>
        """
    return html_content

# Custom CSS
custom_css = """
#main-container {
  display: flex;
  flex-direction: column;
  height: 90vh; /* Adjust overall height as needed */
  width: 100%;
  margin: 0;
  padding: 0;
}

#chat-box {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 10px;
  /* Optional: Add a background if desired */
}

/* Bubbles */
.bubble-container {
  margin: 10px 0;
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.bubble-user {
  align-self: flex-end;
  background-color: #444;
  color: #fff;
  padding: 8px;
  border-radius: 10px;
  max-width: 70%;
  text-align: right;
}

.bubble-answer {
  align-self: flex-start;
  background-color: #333;
  color: #fff;
  padding: 8px;
  border-radius: 10px;
  max-width: 70%;
  position: relative;
}

.feedback-buttons {
  display: inline-block;
  margin-left: 8px;
}

.feedback-text {
  margin-top: 5px;
  font-size: 0.85em;
}

/* Input area pinned near bottom */
#input-area {
  flex: 0 0 auto;
  margin-top: 0; /* Remove extra top margin so it stays near bottom */
  padding: 10px;
}
"""

js_code = """
<script>
function sendFeedback(index, feedback) {
    let feedbackInput = document.querySelector('#feedback_input input');
    feedbackInput.value = index + ',' + feedback;
    let feedbackButton = document.querySelector('#feedback_button button');
    feedbackButton.click();
}
</script>
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# ConfiDent AI")
    
    with gr.Column(elem_id="main-container"):
        conversation = gr.HTML(elem_id="chat-box")
        state = gr.Textbox(value="[]", visible=False)
        
        with gr.Column(elem_id="input-area"):
            user_input = gr.Textbox(
                placeholder="Ask a dental/oral health question...",
                label="Your Message"
            )
            user_input.submit(
                fn=update_history, 
                inputs=[user_input, state], 
                outputs=[user_input, state, conversation]
            )
    
    feedback_input = gr.Textbox(value="", visible=False, elem_id="feedback_input")
    feedback_button = gr.Button("Send Feedback", visible=False, elem_id="feedback_button")
    feedback_button.click(
        fn=handle_feedback, 
        inputs=[feedback_input, state], 
        outputs=[state, conversation]
    )
    
    gr.HTML(js_code)
    
demo.launch()