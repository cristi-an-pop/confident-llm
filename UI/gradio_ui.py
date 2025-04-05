import json
import gradio as gr

def get_answer(user_message):
    return f"This is a dummy answer for: '{user_message}'"

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