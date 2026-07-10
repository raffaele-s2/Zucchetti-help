import streamlit as st
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import google.generativeai as genai


st.set_page_config(
    page_title="Chatbot PDF",
    page_icon="📚"
)


st.title("📚 Assistente documentale PDF")


# API KEY GOOGLE
api_key = st.sidebar.text_input(
    "Google API Key",
    type="password"
)

if api_key:
    genai.configure(api_key=api_key)


# Upload PDF

files = st.file_uploader(
    "Carica i tuoi PDF",
    type="pdf",
    accept_multiple_files=True
)


if files:

    documents=[]

    for file in files:

        path=f"./{file.name}"

        with open(path,"wb") as f:
            f.write(file.getbuffer())


        loader=PyMuPDFLoader(path)

        documents.extend(
            loader.load()
        )


    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )


    chunks=splitter.split_documents(
        documents
    )


    embeddings=HuggingFaceEmbeddings(
        model_name=
        "BAAI/bge-m3"
    )


    db=FAISS.from_documents(
        chunks,
        embeddings
    )


    st.session_state.db=db


    st.success(
        "Documenti indicizzati!"
    )



if "db" in st.session_state:

    domanda=st.chat_input(
        "Fai una domanda sui documenti..."
    )


    if domanda:

        docs=st.session_state.db.similarity_search(
            domanda,
            k=4
        )


        contesto="\n\n".join(
            [
                d.page_content
                for d in docs
            ]
        )


        prompt=f"""
Sei un assistente tecnico.

Rispondi usando solo il CONTENUTO.

Se non trovi la risposta scrivi:
"Informazione non presente nei documenti."

CONTENUTO:

{contesto}

DOMANDA:

{domanda}
"""


        model=genai.GenerativeModel(
            "gemini-2.0-flash"
        )


        risposta=model.generate_content(
            prompt
        )


        st.write(
            risposta.text
        )


        st.caption(
            "Fonti:"
        )


        for d in docs:
            st.write(
                d.metadata
            )
