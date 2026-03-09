import os
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Demo", page_icon=":mag:", layout="wide")
st.title("RAG AI (LangChain + Gemini)")

default_pdf = "what-is-cancer.pdf"
pdf_path = st.sidebar.text_input("PDF path", value=default_pdf)
api_key = st.sidebar.text_input(
    "GOOGLE_API_KEY",
    value=os.getenv("GOOGLE_API_KEY", ""),
    type="password",
    help="Leave blank to use environment variable if already set.",
)
show_debug = st.sidebar.checkbox("Show retrieved chunks", value=True)

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

@st.cache_resource(show_spinner=False)
def build_rag_components(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    loader = PyPDFLoader(path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 10},
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a medical customer support assistant.

Answer only using the provided context.
If the answer is not found in the context, say:
"I couldn't find that in the support knowledge base."

Context:
{context}

Question:
{question}
"""
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return retriever, rag_chain, len(documents), len(splits)


if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Set GOOGLE_API_KEY in sidebar or environment before running queries.")
    st.stop()

try:
    retriever, rag_chain, doc_count, chunk_count = build_rag_components(pdf_path)
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.stop()

st.caption(f"Loaded documents: {doc_count} | Chunks: {chunk_count}")
question = st.text_input("Ask a question", value="when cancer spread and grows?")

if st.button("Run RAG", type="primary"):
    with st.spinner("Retrieving and generating answer..."):
        retrieved_docs = retriever.invoke(question)
        answer = rag_chain.invoke(question)

    st.subheader("Answer")
    st.write(answer)

    if show_debug:
        st.subheader("Retrieved Chunks")
        for idx, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "n/a")
            with st.expander(f"Chunk {idx} | source={source} | page={page}"):
                st.write(doc.page_content)
