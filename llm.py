import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import faiss

st.title("LLM Chatbot News Insights")
st.sidebar.title("News Articles URLs")

# st.set_page_config(page_title = 'News Research Tool')
st.markdown("""
# Get instant insights from online news
# How it works
Follow the instructions

1. Enter your API key
2. Copy Paste the online news links
3. Ask your questions
""")

api_key = st.text_input('Enter your Google API key: ', type='password', key='api_key_input')

# def get_text_from_urls(urls):
#     loader = UnstructuredURLLoader(urls = urls)
#     # main_placeholder.text("Processing URLs...✅")
#     data = loader.load()
#     return data


import requests
from bs4 import BeautifulSoup

def get_text_from_urls(urls):
    all_texts = []
    for url in urls:
        try:
            # Send a GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract all text content from paragraphs and headings
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div'])

            # Concatenate all text content into a single string
            full_text = ' '.join([element.get_text(separator=' ', strip=True) for element in text_elements])

            # Append the extracted text to the list
            all_texts.append(full_text)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from URL {url}: {e}")
            all_texts.append(f"Error: {e}")

    return all_texts


def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        separators = ['\n\n',"\n",'.',' ']
    )
    docs = text_splitter.create_documents(data)
    return docs

def get_vector_store(docs, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model= "models/embedding-001", 
                                              google_api_key = api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local('faiss_index')
    
def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details,
        if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    model = ChatGoogleGenerativeAI(model = 'gemini-pro', temperature = 0.3, google_api_key = api_key)
    prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model= "models/embedding-001", google_api_key = api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ",response['output_text'])

def main():
    st.header("LLM Chatbot News Insights")
    user_question = st.text_input("Ask your question: ", key="user_question")
    if user_question and api_key:
        user_input(user_question, api_key)

    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    # main_placeholder = st.empty()
    process_url_clicked = st.sidebar.button("Process URLs")
    if process_url_clicked and api_key:
        with st.spinner("Processing URLs..."):
            data = get_text_from_urls(urls)
            docs = get_text_chunks(data)
            get_vector_store(docs, api_key)
            st.success("Done")

if __name__ == "__main__":
    main()

# AIzaSyDZ3HyHcDNgwHtytM8oGMXNaj15TtwQfPk



# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)
# process_url_clicked = st.sidebar.button("Process URLs")


# main_placeholder = st.empty()

# if process_url_clicked:
#     loader = UnstructuredURLLoader(urls = urls)
#     main_placeholder.text("Processing URLs...✅")
#     data = loader.load()
    
#     main_placeholder.text("Text Splitter Started...✅")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 1000,
#         separators = ['\n\n',"\n",'.',' ']
#     )

#     docs = text_splitter.split_documents(urls)
#     embeddings = GoogleGenerativeAIEmbeddings(model= "models/embedding-001", 
#                                               google_api_key = api_key)
#     main_placeholder.text("Vector started building...✅")
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     vectorstore.save_local('faiss_index')


# user_question = main_placeholder.text_input("Ask your question: ")
# if user_question:
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)
#     docs = new_db.similarity_search(user_question)

#     prompt_template = """
#         Answer the question as detailed as possible from the provided context, make sure to provide all the details,
#         if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#         Context:\n {context}?\n
#         Question: \n{question}\n

#         Answer:
#         """
#     model = ChatGoogleGenerativeAI(model = 'gemini-pro', temperature = 0.3, google_api_key = api_key)
#     prompt = PromptTemplate(
#             template=prompt_template, input_variables=["context", "question"]
#         )
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     response = chain(
#             {"input_documents": docs, "question": user_question}, return_only_outputs=True
#         )
#     st.subheader("Answer")
#     st.write(response['output_text'])


# def main():
#     st.header("LLM Chatbot News Insights")
#     user_question = st.text_input("Ask your question: ", key="user_question")
#     if user_question and api_key:
#         user_input(user_question, api_key)

#     with st.sidebar:
#         st.title("News Articles URLs")
#         urls = []
#         for i in range(3):
#             url = st.text_input(f"URL {i+1}")
#             urls.append(url)
#         process_url_clicked = st.button("Process URLs")
#         if process_url_clicked and api_key:
#             with st.spinner("Processing URLs..."):
#                 process_urls(urls, api_key)
            