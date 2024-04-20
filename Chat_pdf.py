import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
llm_repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_kwargs = {"temperature": 0.5, "max_length": 250, "max_new_tokens": 500}



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context and your existing knowledge. Provide helpful, accurate, and engaging answers make sure to provide all the details in the suitable markdown format, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    from langchain_community.chat_models import BedrockChat
    model = BedrockChat(model_id="mistral.mixtral-8x7b-instruct-v0:1", model_kwargs={"temperature": 0.6},region_name="us-east-1",credentials_profile_name="default")
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

    return prompt, model



def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})
    
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs = {"k": 10})

    prompt,model = get_conversational_chain()

    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser())


    #print(response)
    st.markdown(f"<div style='background-color:#183c2c; padding: 10px; border-radius: 5px;'><span style='font-size:20px;'>{rag_chain.invoke(user_question)}</span></div>", unsafe_allow_html=True)
    



def main():
    st.set_page_config("Chat PDF")
    st.header("Lets Chat With your PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()