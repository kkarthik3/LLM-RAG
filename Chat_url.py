import os
import streamlit as st
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import BedrockChat
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()  



st.title("News Or Blog Research Tool 📝🗣️💬")
st.sidebar.title("Please Give Input Url to be analysed 🤖🦾")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

with st.sidebar:
    st.divider()
    st.caption("By Karthikeyan 🔥")

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "Link DB/faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = BedrockChat(model_id="mistral.mixtral-8x7b-instruct-v0:1", model_kwargs={"temperature": 0.6},region_name="us-east-1",credentials_profile_name="default")

if process_url_clicked:
    # load data
    loader = SeleniumURLLoader(urls=urls)
    main_placeholder.subheader("The commencement of data loading has been initiated...📊💫🌟")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=10000,chunk_overlap = 1000
    )
    main_placeholder.subheader("The commencement of the Text Splitter's operation has been initiated...📜✂️🌟")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.subheader("The construction of the embedding vector has commenced...🌟🔨🚀")
    with st.sidebar:
        st.warning("Link is Processed", icon="💡")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)


prompt_template = """
Answer the question as detailed as possible from the provided context and your existing knowledge. Provide helpful, accurate, and engaging answers make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
Context:\n {context}?\n
Question: \n{question}\n
Answer:
"""


query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            db = vectorstore.similarity_search(query)
            prompt = PromptTemplate(template =prompt_template, input_variables = ["context", "question"])
            chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
            result = chain({"input_documents":db, "question": query}, return_only_outputs=True)
            result = result["output_text"]
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.markdown(f"<div style='background-color:#183c2c; padding: 10px; border-radius: 5px;'><span style='font-size:20px;'>{result}</span></div>", unsafe_allow_html=True)