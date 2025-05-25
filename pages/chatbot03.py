'''
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

print("All documents have been embedded and uploaded to Pinecone.")'''

#-----------------------


import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import pinecone

# Load environment variables
load_dotenv()

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_KEY = os.getenv("GROQ_KEY")

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
# Example: create an index if it doesn't exist
if 'my_index' not in [index['name'] for index in pc.list_indexes()]:
    pc.create_index(
        name='my_index',
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to Pinecone index
def get_retriever():
    vectorstore = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Build Conversational RAG Chain
def get_conversational_chain():
    llm = ChatGroq(
        groq_api_key=GROQ_KEY,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )
    retriever = get_retriever()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# UI starts here
def main():
    st.set_page_config(page_title="Conversational RAG Chatbot", layout="centered")
    st.title("🧠 RAG Chatbot with Groq + Pinecone")

    # Only allow logged in users
    if "user" not in st.session_state or not st.session_state["user"]:
        st.warning("You must be logged in to use the chatbot. Please log in from the main page.")
        st.stop()

    # Add logout button
    logout_col, _ = st.columns([1, 5])
    with logout_col:
        if st.button("Log Out"):
            st.session_state.pop("user", None)
            st.success("Logged out successfully!")
            st.stop()

    # Get user id for namespace
    user_id = st.session_state["user"].get("localId") or st.session_state["user"].get("email")
    namespace = user_id.lower().replace(" ", "-") if user_id else None

    # Session state setup
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_chain" not in st.session_state or st.session_state.get("rag_chain_namespace") != namespace:
        # Build retriever for this user's namespace
        def get_retriever():
            vectorstore = Pinecone.from_existing_index(
                index_name=PINECONE_INDEX_NAME,
                embedding=embedding_model,
                namespace=namespace
            )
            return vectorstore.as_retriever(search_kwargs={"k": 5})
        def get_conversational_chain():
            llm = ChatGroq(
                groq_api_key=GROQ_KEY,
                model_name="llama3-8b-8192"
            )
            retriever = get_retriever()
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
        st.session_state.rag_chain = get_conversational_chain()
        st.session_state.rag_chain_namespace = namespace

    # Display conversation
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # User input
    user_input = st.chat_input("Ask your question...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_chain({
                        "question": user_input,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = result["answer"]
                    st.session_state.chat_history.append(("assistant", answer))
                    st.markdown(answer)

                    # Optional: Show sources
                    with st.expander("🔍 Sources"):
                        for doc in result["source_documents"]:
                            st.markdown(f"- `{doc.metadata.get('source', doc.metadata.get('filename', 'unknown'))}`")
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.chat_history.append(("assistant", error_msg))
                    st.error(error_msg)

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()
