import gradio as gr
import time

from dotenv import load_dotenv

load_dotenv()


from agent import generate


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history


def bot(message, history):
    # print("Old History:", history)
    # print("Old Message:", message)
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

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot = gr.ChatInterface(fn=bot, type="messages", multimodal=True, save_history=True, flagging_options=["like", "dislike"], flagging_mode="manual")

        with gr.Column():
            gr.File(height=809, label="File Browser", elem_id="file_browser", interactive=True)

if __name__ == "__main__":
    demo.launch()