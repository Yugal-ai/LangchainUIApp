""" This Code is for working with PDF files"""
# import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import  FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import streamlit_authenticator as stauth

def main():
    """ This is main funtion of this program, All execution will work from here """
    load_dotenv()
    st.set_page_config(page_title="Ask you PDF")
    st.header("Ask you PDF")

    pdf = st.file_uploader("Upload you PDF here" , type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator= "\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks= text_splitter.split_text(text)

        # Create Embeddings
        embedding = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embedding)

        user_question = st.text_input("Ask you Question here...")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            # st.write(docs)

            llm = OpenAI()
            chains = load_qa_chain(llm, chain_type="stuff" )
            with get_openai_callback() as cb:
                response=chains.run(input_documents = docs , question = user_question)
                print(cb)
                
            st.write(response)


if __name__ == '__main__':
    main()
