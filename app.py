import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai



from dotenv import load_dotenv
load_dotenv

os.environ['GROQ_API']=os.getenv('GROQ_API')

groq_api_key=os.getenv("GROQ_API")

llm=ChatGroq(groq_api_key=groq_api_key,model_name='Gemma-7b-It')

prompt=ChatPromptTemplate.from_template(
      """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """
)

