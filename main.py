

import os
import streamlit as st
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI

# Get OpenAI API Key securely
oa_token = st.secrets["OA_API_KEY"]

# Streamlit UI
st.title("WebWit")
st.markdown("**WebWit** is your AI-powered research sidekick! Just drop in a couple of URLs, and it’ll read, digest, and let you quiz the content like a pro.")
st.sidebar.title("Input")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_folder = "faiss_index_store"

main_placeholder = st.empty()

# Initialize LLM with GPT-3.5
llm = OpenAI(
    # model_name="gpt-3.5-turbo",
    temperature=0.6,
    max_tokens=400,
    api_key=oa_token
)
# llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load and process data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started..")
    data = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting...Started...")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save to FAISS
    embeddings = OpenAIEmbeddings(api_key=oa_token)
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    # Save FAISS index to disk (no pickle)
    vectorstore_openai.save_local(index_folder)

# Input and answer handling
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_folder):
        embeddings = OpenAIEmbeddings(api_key=oa_token)
        vectorstore = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)

        # Show result
        st.header("Answer")
        st.write(result["answer"])

        # Show sources if any
        sources = result.get("sources", "")
        if sources:
            st.subheader("Source:")
            for source in sources.split("\n"):
                st.write(source)

