WebWit – Your AI-Powered Research Sidekick!
WebWit lets you drop in a couple of URLs and chat with the content like a pro. It scrapes web pages, summarizes them, and allows you to ask intelligent questions—all powered by OpenAI and LangChain.

Features
Input URLs: Paste up to 2 URLs from which you want to extract information.

Automatic Parsing: Web pages are fetched and converted into readable text using the UnstructuredURLLoader.

Chunking and Embedding: Text is chunked using RecursiveCharacterTextSplitter and converted into embeddings using OpenAIEmbeddings.

FAISS Vector Store: Data is stored locally in a FAISS index for efficient semantic retrieval.

QA Interface: Ask questions about the content and receive AI-generated answers with sources using RetrievalQAWithSourcesChain.

Tech Stack
Streamlit — For the UI

LangChain — To chain together LLMs and retrieval logic

FAISS — Local vector store for similarity search

OpenAI GPT-3.5 — The brain behind the operation

