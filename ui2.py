import gradio as gr
import time

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from agent import generate
from langchain_core.messages import SystemMessage, HumanMessage
# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.
llm = init_chat_model("google_genai:gemini-2.0-flash")

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)



def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot(history: list[dict]):
    result = generate(history)
    # Extract the last AI message
    messages = result["messages"]
    ai_msg = messages[-1]  # should be AIMessage
    response_text = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)

    # Append & stream response
    history.append({"role": "assistant", "content": ""})
    for character in response_text:
        history[-1]["content"] += character
        time.sleep(0.05)
        yield history



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(elem_id="chatbot", type="messages", height=740, save_history=True)

            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="Enter message or upload file...",
                show_label=False,
                sources=["microphone", "upload"],
            )

            chat_msg = chat_input.submit(
                add_message, [chatbot, chat_input], [chatbot, chat_input]
            )
            bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

            chatbot.like(print_like_dislike, None, None, like_user_message=True)
        with gr.Column():
            gr.File(height=809, label="File Browser", elem_id="file_browser", interactive=True)

if __name__ == "__main__":
    demo.launch()