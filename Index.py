import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import pickle
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import SeleniumURLLoader
import time

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from pandasai import SmartDataframe
import pandas as pd
import textwrap
from IPython.display import display
from IPython.display import Markdown
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config("RAG with Large Language Models",layout="wide",page_icon="robot")


load_dotenv()
genai.configure(api_key=os.getenv("google_api_key"))

selected2 = option_menu(None, ["Home", "CSV", "Web", 'PDF', "JSON"], 
    icons=['house', 'table', "globe", 'file-pdf',"filetype-json"], 
    menu_icon="cast", default_index=0, orientation="horizontal")


def chat_pdf():
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
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")


    def get_conversational_chain():

        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        return chain


    def user_input(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        print(response)
        st.subheader("Reply: ")
        st.markdown(f"<span style='font-size:20px'>{response['output_text']}</span>", unsafe_allow_html=True)




    def main():
        
        st.header("Engage in a delightful conversation with your PDF...📚💬✨")
        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.subheader("Upload your PDF Files and Click on the Submit & Process Button...📄💻🔄")
            pdf_docs = st.file_uploader("", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing...🔄🔄"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("the task is now complete 🎉🌟")
                    
            st.divider()
            st.caption("By Karthikeyan 🔥")



    if __name__ == "__main__":
        main()


def hyperlink():
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
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.getenv("google_api_key"),temperature=0.8)

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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.subheader("The construction of the embedding vector has commenced...🌟🔨🚀")
        with st.sidebar:
            st.warning("Link is Processed", icon="💡")
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)


    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
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
                # result will be a dictionary of this format --> {"answer": "", "sources": [] }
                st.header("Answer")
                st.markdown(f"<span style='font-size:17px'>{result['output_text']}</span>", unsafe_allow_html=True)
                
            
                sources = result.get("sources", "")
            
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)


def chat_csv():
    def get_image_filename(directory):
        image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            return image_files[0]  # Return the first image file found
        else:
            return None
        
    load_dotenv()
    #st.set_page_config(page_title="MRI Image Super Resolution", layout="wide")

    llm = ChatGoogleGenerativeAI(model="models/gemini-pro", google_api_key=os.getenv("google_api_key"),
                                temperature=1,convert_system_message_to_human=True)

    def to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

    st.title("Chat With Your Dataset 📝🗣️💬")
    st.header("Exploratory Data Analysis 🔎📉📊")

    # Reset cache function
    def clear_cache():
        st.cache_data.clear()

    with st.sidebar:
        st.title("Your AI Assistant here 🤖🦾")
        if 'clicked' not in st.session_state:
            st.session_state.clicked = {1: False}
        
        def clicked(button):
            st.session_state.clicked[button] = True
        
        csv = st.file_uploader("Upload your Csv File here", type=["csv", "xlsx"])
        if st.button("Submit & Process", on_click=clicked, args=[1]):
            with st.spinner("Processing..."):
                if csv is not None:
                    csv.seek(0)
                    df = pd.read_csv(csv, low_memory=False, encoding='latin-1')
                    st.session_state.df = df

                    with st.expander("Steps of EDA"):
                        eda = llm.invoke("What are the steps of EDA")
                        st.write(eda.content)

        # Add reset cache button
        st.sidebar.button("Refresh",on_click=clear_cache)

        
        st.divider()
        st.caption("By Karthikeyan 🔥")
        
    @st.cache_data()
    def domain(query):   
        if query:
            with st.expander("Domain Knowledge"):
                domain = llm.invoke(query)
                st.write(domain.content)
        return


    prompt = "Don't provide any python codes analyse only what do you know about this dataset"
    @st.cache_data
    def function_agent1():
        st.subheader("Data Overview")
        st.write("The sample of the dataset Look like This")
        st.write(st.session_state.df.sample(5))
        st.subheader("Meaning of the coloumns")
        st.write(agent.run("What are the meaning of the columns tells about the dataset explain them in order by order"+prompt))
        st.subheader("Missing values")
        st.write(agent.run("How many missing values does this dataframe have? if yes Start the answer with 'There are'"+prompt))
        st.subheader("Duplpicate Values")
        st.write(agent.run("Are there any duplicate values and if yes what are the most duplicate entries?, if yes show in order"+prompt))
        return

    @st.cache_data
    def function_agent2():   
        st.subheader("Data Summarization")
        st.write(st.session_state.df.describe())
        st.subheader("Correlation Analysis")
        st.write(df2.chat("Calculate correlations between numerical variables to identify potential relationships. and don't include Nan values, i dont wnat ay python codes"))
        st.subheader("Outlier analysis")
        st.write(agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis."+prompt))
        st.subheader("Feature Engineering")
        st.write(agent.run("What new features would be interesting to create?, and how to create that and tell the logic behind them."+prompt))
        return


    def question():
        st.subheader("Is there any Questions regarding to this dataframe🤔❓")
        data = st.text_input("Enter your Question",key = "1")
        if data:
            st.write(df2.chat(data +  " i dont want code only output with sentence"))
        return

    with st.sidebar:
        st.subheader("Any Questions about domain knowledge 🤔❓")
        query = st.text_input("Enter your Question",key="2")
        domain(query)
    

    def plot(query):
        file_path = r"exports\charts"
        image_filename = get_image_filename(file_path)
        if image_filename:
            image_path = os.path.join(file_path, image_filename)
            if os.path.exists(image_path):
                os.remove(image_path)
        if query:
            df2.chat(query)
            st.pyplot()
            # image_filename = get_image_filename(file_path)
            # if image_filename:
            #     image_path = os.path.join(file_path, image_filename)
            #     if image_path:
            #         st.image(image_path)
        return


    # main
    if 'df' in st.session_state:
        agent = create_pandas_dataframe_agent(llm=llm, df=st.session_state.df,agent_type="openai-tools",verbose = True)
        df2 = SmartDataframe(df=st.session_state.df, config={"llm": llm,"open_charts":False,"save_charts":True})
        function_agent1()
        function_agent2()
        st.divider()
        question()
        query2 = st.text_input("Give query to plot",key="3")
        if query2:
            plot(query2)


def chat_json():
    load_dotenv()
    #st.set_page_config(page_title="MRI Image Super Resolution", layout="wide")

    llm = ChatGoogleGenerativeAI(model="models/gemini-pro", google_api_key=os.getenv("google_api_key"),
                                temperature=1,convert_system_message_to_human=True)


    def get_image_filename(directory):
        image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            return image_files[0]  # Return the first image file found
        else:
            return None

    def to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

    st.title("Chat With Your Json File 📝🗣️💬")

    # Reset cache function
    def clear_cache():
        st.cache_data.clear()

    with st.sidebar:
        st.title("Your AI Assistant here 🤖🦾")
        if 'clicked' not in st.session_state:
            st.session_state.clicked = {1: False}
        
        def clicked(button):
            st.session_state.clicked[button] = True
        
        json = st.file_uploader("Upload your Json File Here", type=["json"])
        if st.button("Submit & Process", on_click=clicked, args=[1]):
            with st.spinner("Processing..."):
                if json is not None:
                    json.seek(0)
                    dfjson = pd.read_json(json, encoding='latin-1')
                    st.session_state.dfjson = dfjson

        # Add reset cache button
        st.sidebar.button("Refresh",on_click=clear_cache)

        st.divider()
        st.caption("By Karthikeyan 🔥")

    if 'dfjson' in st.session_state:
        chatmodel = SmartDataframe(df=st.session_state.dfjson, config={"llm": llm,"open_charts":False,"save_charts":True})
        query = st.text_input("Ask your question regarding json file")

        if query:
            answer = chatmodel.chat(query)
            if answer:
                st.write("The Answer is:")
                st.write(answer)



def home():
    import streamlit as st

    st.write("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        font-size: 35px;
    }
             
    img {
        width: 1000px; /* Adjust the width as needed */
        height: auto; /* Maintain aspect ratio */
        display: block; /* Centerize the image */
        margin-left: 280px;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    st.write('<div class="center"><h1>Welcome to the AI Base RAG for Your Data! 🤖🎉</h1></div>', unsafe_allow_html=True)
    st.markdown("<span style='font-size:30px;display: flex;justify-content: center;'><b>[Made by Karthikeyan K ](https://github.com/kkarthik3)</b></span>",unsafe_allow_html=True)

    st.image("LLM.jpg")


if selected2 == "PDF":
    chat_pdf()
elif selected2 == "Web":
    hyperlink()
elif selected2 =="CSV":
    chat_csv()
elif selected2 == "JSON":
    chat_json()
elif selected2 == "Home":
    home()
