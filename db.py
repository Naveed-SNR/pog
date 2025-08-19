import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os
import gradio as gr

load_dotenv()

client = chromadb.PersistentClient("./chroma_langchain_db")
collection = client.get_collection("ragtag")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma(
    collection_name="ragtag",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

def preview_collection():
    results = collection.get(include=["metadatas"])
    results = collection.peek()
    print(results)

def add_to_chroma(files):
    for file in files:
        filename = os.path.basename(file)
        print(f"Processing file: {file}")
        loader = PyPDFLoader(
            file,   
            mode="single",
            extraction_mode="plain"
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        for chunk in all_splits:
            chunk.metadata["source"] = filename
        vector_store.add_documents(all_splits)  # Save changes to the vector store
    print(f"✅ Ingested {len(files)} file(s) into ChromaDB.")

def remove_from_chroma(delete_data: gr.DeletedFileData):
    filename = os.path.basename(delete_data.file.path)
    collection.delete(where={"source": filename})
    print(f"✅ Removed {filename} from ChromaDB.")