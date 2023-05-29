""" This Code is for working with PDF files"""
# import os
import sys
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import  FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# user_question = st.text_input("Ask you Question for doc...")

def check_api_key(api_key):
    """This function is to check whether given API key is working or not"""
    openai.api_key = api_key
    try:
        # Make a test API call to check if the key is valid
        response = openai.Completion.create(engine='davinci', prompt='Hello', max_tokens=5)
        return True
          
    except Exception as excep:
        print("Error Occured " , excep)
        return False
     
def generate_prompt(prompt):
    """This function will address queries outside the world"""
    # openai.api_key = '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'
   
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100
    )
    completion = response.choices[0].text.strip()
    return completion
                    
# Call the function to generate a prompt

def split_embed_single(pdf):
    """ Purpose of this function is to Split and Embed Sigle CV """
    chain_type = st.selectbox('Select your required Chain Type', 
                            ('stuff', 'map_reduce', 'refine', 'map_rerank'))
    user_question = st.text_input("Ask you Question for doc...")

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

    # user_question = st.text_input("Ask you Question here...")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        # st.write(docs)

        llm = OpenAI()
        chains = load_qa_chain(llm, chain_type=chain_type)
        with get_openai_callback() as cb:
            response=chains.run(input_documents = docs , question = user_question)
            print(cb)     
        st.write(response)

def split_embed_two(lst):
    """ Purpose of this function is to Split and Embed Two CV """
    chain_type = st.selectbox('Select your required Chain Type', 
                            ('stuff', 'map_reduce', 'refine', 'map_rerank'))
    user_question = st.text_input("Ask you Question for doc...")

    embedding = OpenAIEmbeddings()
    knowledge_base = None

    for pdf in lst:
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
        
        knowledge_base = knowledge_base + FAISS.from_texts(chunks, embedding)

    # user_question = st.text_input("Ask you Question here...")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        # st.write(docs)

        llm = OpenAI()
        chains = load_qa_chain(llm, chain_type=chain_type)
        with get_openai_callback() as cb:
            response=chains.run(input_documents = docs , question = user_question)
            print(cb)     
        st.write(response)


def main():
    """ This is main funtion of this program, All execution will work from here """
    load_dotenv()
    st.set_page_config(page_title="Ask you PDF")
    st.header("Ask you PDF")

    api_key = st.text_input("Kindly enter your API key here")
    res = check_api_key(api_key)
    if res:
        st.success('Your key is working fine you can proceed further', icon="✅")
        option = st.radio("Comparision Option", ('Single Candidate' , 'Two Candidate'))
        if option == 'Single Candidate':
            candi_pdf = st.file_uploader("Upload you PDF here" ,  type="pdf")
            if candi_pdf is not None:
                split_embed_single(candi_pdf)
        elif option == 'Two Candidate':
            candi_pdf_a = st.file_uploader("Upload you PDF for Candidate_1" , type="pdf")
            candi_pdf_b = st.file_uploader("Upload you PDF for Candidate_2" , type="pdf")
            if candi_pdf_a and candi_pdf_b is not None:
                lst = list(candi_pdf_a,candi_pdf_b)
                split_embed_two(lst)

    else:
        st.error('Seems some issue with your API Key', icon="🚨")
        sys.exit()

    

    # chain_type = st.selectbox('Select your required Chain Type', 
    #                             ('stuff', 'map_reduce', 'refine', 'map_rerank'))

    

    # toggle_button = st.checkbox("Global Data")

   

    # if not toggle_button:
    #     # chain_type = st.selectbox('Select your required Chain Type', 
    #     #                         ('stuff', 'map_reduce', 'refine', 'map_rerank'))
        
    #     user_question = st.text_input("Ask you Question for doc...")

    #     if pdf is not None:
    #         pdf_reader = PdfReader(pdf)
    #         text = ""
    #         for page in pdf_reader.pages:
    #             text+= page.extract_text()

    #         # Split into chunks
    #         text_splitter = CharacterTextSplitter(
    #             separator= "\n",
    #             chunk_size=1000,
    #             chunk_overlap=200,
    #             length_function=len
    #         )

    #         chunks= text_splitter.split_text(text)

    #         # Create Embeddings
    #         embedding = OpenAIEmbeddings()
    #         knowledge_base = FAISS.from_texts(chunks, embedding)

    #         # user_question = st.text_input("Ask you Question here...")
    #         if user_question:
    #             docs = knowledge_base.similarity_search(user_question)
    #             # st.write(docs)

    #             llm = OpenAI()
    #             chains = load_qa_chain(llm, chain_type=chain_type)
    #             with get_openai_callback() as cb:
    #                 response=chains.run(input_documents = docs , question = user_question)
    #                 print(cb)     
    #             st.write(response)
    # elif toggle_button:
    #     prompt = st.text_input("Enter your prompt for Global Data ")
    #     prompt_output = generate_prompt(prompt)
    #     st.write(prompt_output)

    # else:
    #     st.write("Kindly Choose one of the Option to facilitate you")

if __name__ == '__main__':
    main()
