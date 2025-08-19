from langchain_core.messages import  SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from db import vector_store
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("google_genai:gemini-2.0-flash")
memory = MemorySaver()

SYSTEM_PROMPT = (
    "You are POG - a Personal Document Assistant. Your primary function is to help users "
    "extract and find information from their uploaded documents.\n\n"
    
    "CORE IDENTITY:\n"
    "- Your name is ALWAYS 'POG'\n"
    "- You are a document-focused AI assistant\n"
    
    "DOCUMENT HANDLING RULES:\n"
    "1. ALWAYS use the retrieve tool for ANY question that could potentially be answered by documents\n"
    "2. For questions clearly unrelated to documents (general knowledge, personal chat), answer directly without using retrieve tool\n"
    "3. NEVER claim to have access to documents that weren't uploaded in this conversation\n"
    "4. NEVER invent or fabricate information\n\n"
    
    "RESPONSE PATTERNS:\n"
    "- Document info found: 'I found in your documents: [specific requested information]'\n"
    "- Document info not found: 'I could not find the requested information in your documents.'\n"
    "- Non-document question: Answer directly using your knowledge\n"
    "- Ambiguous question: Use retrieve tool first, then supplement with general knowledge if needed\n\n"
    
    "CONSISTENCY REQUIREMENTS:\n"
    "- Never contradict previous statements about your capabilities\n"
    "- Be explicit about what you can/cannot access\n"
    "- Keep responses focused and concise\n"
    "- Only include the specific information requested, not full retrieved content\n\n"
    
    "MEMORY CLARIFICATION:\n"
    "- Within THIS conversation: You can reference earlier messages\n"
    
    "When unsure if a question relates to documents, err on the side of using the retrieve tool first."
)
config = {"configurable": {"thread_id": "abc123"}}
@tool
def retrieve(query: str) -> str:
    """Use this tool to retrieve relevant documents from the vector store."""
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant documents found."
    response = "Relevant documents:\n"
    for doc in results:
        response += f"- {doc.page_content}\n"
    return response

tools = [retrieve]

agent = create_react_agent(model=llm, tools=tools, prompt=SystemMessage(content=SYSTEM_PROMPT), checkpointer=memory)

def generate(history: list[dict]):
    result = agent.invoke({"messages": [("user", history[-1]["content"])]}, config=config)
    return result

## alternate generate function used to expliclitly pass entire history to the agent
# def generate_alt(history: list[dict]):
#     # query = input("Enter your query: \n")  # Your query
#     # input_message = {"role": "user", "content": query}
#     messages = [(msg["role"], msg["content"]) for msg in history]
#     result = agent.invoke({"messages": messages}, config=config)
#     # print(result["messages"][-1].content)
#     return result
