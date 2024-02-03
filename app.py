from io import BytesIO
import streamlit as st
import shutil
import requests
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def getpdfdoc():
    with st.spinner("Loading PDF..."):
        filename = '48lawsofpower.pdf'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                pdf_doc = f.read()
            return pdf_doc
        else:
            url = 'https://pgcag.files.wordpress.com/2010/01/48lawsofpower.pdf'
            response = requests.get(url)

            with open(filename, 'wb') as f:
                f.write(response.content)
            
            return getpdfdoc()

    
def extract_text_from_pdf(pdf_file_obj):
    with st.spinner("Extracting text from PDF..."):
        pdf_reader = PdfReader(BytesIO(pdf_file_obj))
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page_obj = pdf_reader.pages[page_num]
            text += page_obj.extract_text()
        return text

def get_text_chunks(text):
    with st.spinner("Splitting text into chunks..."):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks


def get_vectorstore(text_chunks):
    with st.spinner("Creating vectorstore..."):
        metadatas = [{"source": f"{i}-pl"} for i in range(len(text_chunks))]
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings, persist_directory="./chroma_db", metadatas=metadatas)
        return vectorstore

def get_conversation_chain(vectorstore):
    with st.spinner("Loading LLM..."):
        llm = ChatOpenAI()

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

def retrain_model():
    st.session_state.conversation = None
    st.session_state.chat_history = None
    pdf_doc = getpdfdoc() # get pdf
    raw_text = extract_text_from_pdf(pdf_doc) # get pdf text
    text_chunks = get_text_chunks(raw_text) # get the text chunks    
    vectorstore = get_vectorstore(text_chunks) # create vector store
    st.session_state.conversation = get_conversation_chain(vectorstore) # create conversation chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown("**User:**")
            st.markdown(message.content)
        else:
            st.markdown("**AI:**")
            st.markdown(message.content)


def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if st.session_state.conversation is None:
        if os.path.isdir("./chroma_db"):
            if os.listdir("./chroma_db"):
                with st.spinner("Loading vector store..."):
                    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
                    st.session_state.conversation = get_conversation_chain(vectorstore)
            else:
                retrain_model()
        else:
            retrain_model()

    if st.session_state.conversation is not None:
        st.sidebar.button("Retrain model", on_click=retrain_model)
        st.header("Ask questions from 48 Laws of Power:books:")
        user_question = st.chat_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()