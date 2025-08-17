from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph, START, END
from IPython.display import Image, display
from dotenv import load_dotenv
load_dotenv()
# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image,audio, & video). Plus shows support for streaming text.
llm = init_chat_model("google_genai:gemini-2.0-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
file_path = "./data.pdf"
loader = PyPDFLoader(
    file_path,
    mode="single",
    extraction_mode="plain"
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally,remove if not necessary
)
vector_store.add_documents(all_splits)
SYSTEM_PROMPT = (
    "You are POG - a Personal Document Assistant. Your goal is to help usersextract and find information from their own uploaded documents. Key rules:\n"
    "1. YOUR NAME IS ALWAYS 'POG'\n"
    "2. You MUST call retrieve tool for EVERY document related question. For questions not related to any documents, don't use the retrieve tool and answer directly using your own knowledge\n"
    "3. You MUST NEVER invent answers\n"
    "4. If you find the requested information, Respond: 'I found in your documents:[content]'"
    "5. If you cannot find the information, Respond: 'I could not find therequested information in your documents.'\n"
    "6. Keep your responses concise and focused on the user's query.\n"
    "7. Don't include all the retrieved content or metadata in your response, only the information that is specifically requested by the user.\n"
    
)
@tool(response_format="content_and_artifact")
def retrieve(query: str) -> str:
    """Use this tool to retrieve relevant documents from the vector store."""
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant documents found."
    
    response = "Relevant documents:\n"
    for doc in results:
        response += f"- {doc.page_content}\n"
    
    return response
tools = ToolNode([retrieve])
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
        # Inject system prompt here
    messages_with_system = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]
    ]
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(messages_with_system)
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}
def generate(state: MessagesState):
    """Generate a response based on retrieved documents (text or OCR from images)"""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    # Relevant, role-specific system prompt
    system_prompt = (  # SIMPLIFIED PROMPT
        SYSTEM_PROMPT + "\n\nRetrieved content:\n" + docs_content
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.type == "tool")
    ]
    prompt = [SystemMessage(content=system_prompt)] + conversation_messages
    response = llm.invoke(prompt)  # Uses the main bound ChatOllama model
    return {"messages": [response]}
# %%
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("tools", tools)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "query_or_respond")
# graph_builder.add_edge("query_or_respond", "tools")
graph_builder.add_conditional_edges(
   "query_or_respond",
   tools_condition,
   {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()
prompt = "What are some of the topics in the second unit?"
result = graph.invoke(
    {"messages": [("user", prompt)]},
)
print(result["messages"][-1].content)
display(Image(graph.get_graph().draw_mermaid_png()))