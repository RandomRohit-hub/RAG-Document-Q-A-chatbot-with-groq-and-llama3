import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
groq_api_key = os.getenv("GROQ_API_KEY")

# Validate GROQ API Key
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in .env file.")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
<context>
{context}
</context>
Question: {input}
""")

st.title("📄 RAG Q&A: attention.pdf + llm.pdf")

# User input
user_prompt = st.text_input("🔍 Enter your query about the PDFs:")

# Vector embedding function
def create_vector_embedding():
    try:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load specific PDFs only
        files = ["attention.pdf", "llm.pdf"]
        loaded_docs = []
        for path in files:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                loaded_docs.extend(loader.load())
            else:
                st.warning(f"⚠️ File not found: {path}")

        if not loaded_docs:
            st.error("❌ No valid documents loaded. Make sure both attention.pdf and llm.pdf exist.")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(loaded_docs)

        if not split_docs:
            st.error("⚠️ Document splitting failed.")
            return

        st.session_state.vectors = FAISS.from_documents(split_docs, st.session_state.embeddings)
        st.session_state.final_documents = split_docs
        st.success("✅ Vector database built from attention.pdf and llm.pdf.")
    except Exception as e:
        st.error(f"🚫 Error creating vector store: {str(e)}")

# Trigger embedding
if st.button("📄 Generate Embeddings from attention.pdf + llm.pdf"):
    create_vector_embedding()

# Q&A section
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please generate embeddings first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed = time.time() - start

        st.subheader("💬 Answer")
        st.write(response.get('answer', 'No answer generated.'))

        st.caption(f"⏱️ Time taken: {round(elapsed, 2)} seconds")

        with st.expander("📘 Similar Document Passages"):
            for i, doc in enumerate(response.get('context', [])):
                st.markdown(f"**Passage {i + 1}:**")
                st.write(doc.page_content)
                st.write("---")
