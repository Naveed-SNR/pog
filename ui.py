import gradio as gr
import time
from dotenv import load_dotenv
from agent import generate
from db import add_to_chroma, remove_from_chroma
load_dotenv()


custom_css = """
html, body, #root, .gradio-container, main.fillable {
    margin: 0 ;
    margin-left: 3px  !important;
    padding: 0 !important;
    height: 100%;
    width: 100%;
    max-width: 100% !important;
}
"""


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history

def bot(message, history):
    history = add_message(history, message)
    result = generate(history)
    
    # Extract the last AI message
    messages = result["messages"]
    ai_msg = messages[-1]  # should be AIMessage
    response_text = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)

    # Append & stream response
    history.append({"role": "assistant", "content": ""})
    for character in response_text:
        history[-1]["content"] += character
        time.sleep(0.01)
        yield history[-1]

with gr.Blocks( css_paths="./style.css", css=custom_css) as demo:
    with gr.Row(elem_classes=["main-row"]):
        with gr.Column(elem_classes=["chat-column"]):
            chatbot = gr.ChatInterface(fn=bot, type="messages", multimodal=True, save_history=True, flagging_options=["like", "dislike"], flagging_mode="manual")
        with gr.Column(elem_classes=["file-column"]):
            file_upload = gr.Files(height=809, label="File Browser", elem_id="file_browser", interactive=True)
            file_upload.upload(
                fn=add_to_chroma,
                inputs=file_upload,
                outputs=None,
            )
            file_upload.delete(
                fn=remove_from_chroma,
                inputs=None,
                outputs=None,
            )

if __name__ == "__main__":
    demo.launch(share=False)
