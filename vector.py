import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone client
pc = Pinecone(api_key=API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-east-1")

# Define the index name
index_name = "user-data-index"

# Create the index if it doesn't exist
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec=spec)

# Connect to the index
index = pc.Index(index_name)

# Define the data directory
data_dir = r"F:\flipr\data"

# Process each user folder
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        docs = []
        metadatas = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append(text)
                    metadatas.append({"filename": filename, "folder": folder})
        if docs:
            # Use the folder name as the namespace
            namespace = folder.lower().replace(" ", "-")
            vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
            vectorstore.add_texts(docs, metadatas=metadatas)

print("All documents have been embedded and uploaded to Pinecone.")
