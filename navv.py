import chromadb 
chroma_client = chromadb.PersistentClient("./chroma_langchain_db")
chroma_collection = chroma_client.create_collection("ragtag")