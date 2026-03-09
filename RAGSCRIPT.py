import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#if not os.getenv("GOOGLE_API_KEY"):
    #raise RuntimeError("GOOGLE_API_KEY is missing. Set it in .env as GOOGLE_API_KEY=your_key")

#os.environ["GOOGLE_API_KEY"] = ""

# 1. Load documents
#files = ["faq.pdf", "policy.pdf", "support_guide.pdf"]
files = ["what-is-cancer.pdf"]
documents = []

for file in files:
    loader = PyPDFLoader(file)
    documents.extend(loader.load())

# 2. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = splitter.split_documents(documents)
#print(f"\nTotal chunks: {len(splits)}")
#for i, d in enumerate(splits, 1):
    #print(f"\n--- CHUNK {i} ---")
    #print(f"source={d.metadata.get('source')} page={d.metadata.get('page')}")
    #print(d.page_content[:500])  # preview

# 3. Create embeddings
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 4. Store in vector DB
vectorstore = FAISS.from_documents(splits, embeddings)

# 5. Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 10}
)

# 6. Prompt
prompt = ChatPromptTemplate.from_template("""
You are a medical customer support assistant.

Answer only using the provided context.
If the answer is not found in the context, say:
"I couldn't find that in the support knowledge base."

Context:
{context}

Question:
{question}
""")

# 7. LLM
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
# 8. Helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 9. RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 10. Ask
question = "when cancer spread and grows?"
retrieved_docs = retriever.invoke(question)

#enable below to see the chunks
#print(f"\nRetrieved docs: {len(retrieved_docs)}")
#for i, d in enumerate(retrieved_docs, 1):
    #print(f"\n--- RETRIEVED {i} ---")
    #print(f"source={d.metadata.get('source')} page={d.metadata.get('page')}")
    #print(d.page_content[:800])


answer = rag_chain.invoke(question)
#print("Retrieved Documents:")
#print("Context:")
print(answer)

