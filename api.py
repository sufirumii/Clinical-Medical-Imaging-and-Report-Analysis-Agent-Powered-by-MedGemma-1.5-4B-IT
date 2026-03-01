
import gradio as gr
from model import run_inference, load_model
import time

load_model()

def chat(message, image, history):
    if not message.strip():
        return history, history, ""
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "Analyzing..."})
    yield history, history, ""
    t0 = time.time()
    result = run_inference(message, image)
    elapsed = time.time() - t0
    history[-1] = {"role": "assistant", "content": result + f"\n\n_(took {elapsed:.1f}s)_"}
    yield history, history, ""

def clear():
    h = [{"role": "assistant", "content": "Cleared. Upload an image and ask me anything."}]
    return h, [], None, ""

with gr.Blocks(title="MedGemma") as demo:
    gr.Markdown("# 🩺 MedGemma 1.5 — Medical AI")
    gr.Markdown("Upload a medical image and ask a question. Supports X-ray, CT, MRI, dermatology, histopathology, lab reports.")

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(label="Upload Medical Image", type="pil")
            gr.Markdown("**Quick Prompts**")
            with gr.Row():
                b1 = gr.Button("Describe X-ray")
                b2 = gr.Button("Key Findings")
            with gr.Row():
                b3 = gr.Button("Differentials")
                b4 = gr.Button("Assess Severity")
            with gr.Row():
                b5 = gr.Button("Full Report")
                b6 = gr.Button("Lab Report")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "👋 Hello! I am MedGemma 1.5. Upload a medical image and ask me anything."}],
                label="MedGemma Analysis",
                height=500,
            )
            msg_in = gr.Textbox(
                placeholder="Type your medical question here...",
                label="Your Question",
                lines=2,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                send_btn  = gr.Button("Analyze →", variant="primary", scale=2)

    state = gr.State([])

    send_btn.click(fn=chat, inputs=[msg_in, image_in, state], outputs=[chatbot, state, msg_in])
    msg_in.submit(fn=chat, inputs=[msg_in, image_in, state], outputs=[chatbot, state, msg_in])
    clear_btn.click(fn=clear, outputs=[chatbot, state, image_in, msg_in])

    b1.click(fn=lambda: "Describe this X-ray in detail", outputs=msg_in)
    b2.click(fn=lambda: "List all key findings and abnormalities", outputs=msg_in)
    b3.click(fn=lambda: "Suggest the top differential diagnoses", outputs=msg_in)
    b4.click(fn=lambda: "Assess severity and clinical urgency", outputs=msg_in)
    b5.click(fn=lambda: "Generate a complete structured radiology report", outputs=msg_in)
    b6.click(fn=lambda: "Summarize this lab report and flag abnormal values", outputs=msg_in)

demo.launch(share=True, show_error=True, theme=gr.themes.Soft(primary_hue="cyan"))
