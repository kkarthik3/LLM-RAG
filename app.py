import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pickle
from langchain_community.document_loaders import SeleniumURLLoader
import time
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from pandasai import SmartDataframe
import pandas as pd
from pandasai.responses.response_parser import ResponseParser
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
import io
import csv

# st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config("RAG with Large Language Models",layout="wide",page_icon="🤖")


load_dotenv()




selected2 = option_menu(None, ["Home", "CSV", "Web", 'PDF', "JSON", "DataBase"], 
    icons=['house', 'table', "globe", 'file-pdf',"filetype-json","database-fill"], 
    menu_icon="cast", default_index=0, orientation="horizontal")


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

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
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")


    def get_conversational_chain():

        prompt_template = """
        Answer the question as detailed as possible from the provided context and your existing knowledge. Provide helpful, accurate, and engaging answers make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        model = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature": 0.5})
        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

        return prompt, model

    def user_input(user_question):
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        
        
        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever(search_kwargs = {"k": 5})

        prompt,model = get_conversational_chain()

        rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser())
        with st.spinner():
            st.subheader("The Answer is")
            st.markdown(f"<div style='background-color:#183c2c; padding: 10px; border-radius: 5px;'><span style='font-size:20px;'>{rag_chain.invoke(user_question)}</span></div>", unsafe_allow_html=True)




    def main():
        
        st.header("Engage in a delightful conversation with your PDF...📚💬✨")
        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.subheader("Upload your PDF Files and Click on the Submit & Process Button...📄💻🔄")
            st.session_state.pdf_docs = st.file_uploader("", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing...🔄🔄"):
                    raw_text = get_pdf_text( st.session_state.pdf_docs)
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
    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature": 0.5})
    
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


def chat_csv():
    def get_image_filename(directory):
        image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            return image_files[0]  # Return the first image file found
        else:
            return None
        
    load_dotenv()

    llm = ChatGoogleGenerativeAI(model="models/gemini-pro", google_api_key=st.session_state.gemini,
                                temperature=1,convert_system_message_to_human=True)



    st.title("Chat With Your Dataset 📝🗣️💬")
    st.header("Exploratory Data Analysis 🔎📉📊")

    # Reset cache function
    def clear_cache():
        st.cache_data.clear()
        st.session_state.clear()

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
                    df = pd.read_csv(csv)
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
        st.write(agent.run("What are the meaning of the columns tells about the dataset explain all of them"+prompt))
        st.subheader("Missing values")
        st.write(agent.run("How many missing values does this dataframe have? if yes Start the answer with 'There are'"))
        st.subheader("Duplpicate Values")
        st.write(agent.run("Are there any duplicate values and if yes what are the most duplicate entries?, if yes show in order"+prompt))
        return

    @st.cache_data
    def function_agent2():   
        st.subheader("Data Summarization")
        st.write(st.session_state.df.describe())
        st.subheader("Correlation Analysis")
        st.write(agent.run("Calculate correlations between numerical variables to identify potential relationships. and don't include Nan values, i dont wnat ay python codes"))
        st.subheader("Outlier analysis")
        st.write(agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis."+prompt))
        st.subheader("Feature Engineering")
        st.write(agent.run("if there are any possible additional features that could be derived from the existing data. Furthermore, explain the underlying logic behind the creation of these potential features"))
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
        if query:
            df2.chat(query)
        return


    # main
    if 'df' in st.session_state:
        agent = create_pandas_dataframe_agent(llm=llm, df=st.session_state.df,verbose = True,agent_type=AgentType.OPENAI_FUNCTIONS,agent_executor_kwargs={'handle_parsing_errors': True})
        df2 = SmartDataframe(df=st.session_state.df, config={"llm": llm,"open_charts":False,"save_charts":True,"response_parser": StreamlitResponse})
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

    llm = ChatGoogleGenerativeAI(model="models/gemini-pro", google_api_key=st.session_state.gemini,
                                temperature=1,convert_system_message_to_human=True)


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



def Chat_db():

    def init_database(user: str, password: str, host: str, port: str, database: str) :
        db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        db = SQLDatabase.from_uri(db_uri)
        return db
    
    llm = ChatGoogleGenerativeAI(model="models/gemini-pro", google_api_key=st.session_state.gemini,
                                temperature=1,convert_system_message_to_human=True)

    def get_query(db):
        template = """I want you to act as a sql developor and coder who write sql queries without syntax error. My first request is "I need you to create a sql query with syntax error". My first suggestion is "You should write a query with a syntax error."
        Based on the table schema below, write a SQL query that would answer the user's question:
        if multiple queries are there write only one single SQL Query without syntax error not anything else
        {schema}

        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

        eg query to write and dont use the back slash(\) anywhere else
        'SELECT COUNT(*) FROM t_shirts;'
        not like this 
        'SELECT COUNT(\\*) FROM t\\_shirts;' 
        Question: {question}

        Question: {question}
        SQL Query:
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
                ("human", template),
            ]
        )

        def get_schema(_):
            return db.table_info.replace('/*', '').replace('*/', '').replace('\n\n', '\n').replace("\n","").replace("\\","").replace("\t","").replace("\\","")

        sql_response = (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
        )
        return sql_response


    def answering(user_question,db):
        template = """You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.:
        {schema}

        Question: {question}
        SQL Query: {query}
        SQL Response: {response}"""
        sql_response = get_query(db)
        prompt_response = ChatPromptTemplate.from_template(template)

        full_chain = (RunnablePassthrough.assign(query=sql_response).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars:db.run(vars["query"]),
        )
        | prompt_response
        | llm
        )

        response = full_chain.invoke({"question": user_question}).content
        return response

    def str_csv(data):
        rows = data.strip('[]()\n').split('), (')
        csv_data = io.StringIO()
        csv_writer = csv.writer(csv_data)
        for row in rows:
            csv_writer.writerow(row.split(', '))
        csv_data.seek(0)
        df = pd.read_csv(csv_data,index_col=None)
        return df

    with st.sidebar:
        st.subheader("Set Your Database Configurations")
        
        host = st.text_input("Host", value="localhost", key="Host")
        port = st.text_input("Port", value="3306", key="Port")
        user = st.text_input("User", value="root", key="User")
        password = st.text_input("Password", type="password", value="admin", key="Password")
        database = st.text_input("Database", value="atliq_tshirts", key="Database")
        
        if st.button("Connect"):
            with st.spinner("Connecting to database..."):
                st.session_state.db = init_database(user, password, host, port, database)
                if st.session_state.db:
                    st.success("Connected to database! 🎉🎉")
        
        def clear_cache():
            st.session_state.clear()
        st.sidebar.button("Refresh",on_click=clear_cache)
        st.divider()
        st.write("Made By Karthikeyan K")





    if 'db' in st.session_state:
        Query = st.text_input("Enter the Qeuries or Question from Your Database:")
        if Query :
            with st.spinner("The SQL Query for your prompt is loading...."):
                sql_query = get_query(st.session_state.db)
                sql_response = sql_query.invoke({"question": f"{Query}"})
                st.markdown(f"<div style='background-color:#183c2c; padding: 10px; border-radius: 5px;'><span style='font-size:20px;'>{sql_response} </span></div>", unsafe_allow_html=True)
            with st.spinner("The Data is loading ...."):
                data = st.session_state.db.run(f"{sql_response}")
                st.dataframe(str_csv(data))
            with st.spinner("Your Query is Performing"):
                response  = answering(Query, st.session_state.db)
                st.markdown(f"<div style='background-color:#183c2c; padding: 10px; border-radius: 5px;'><span style='font-size:20px;'>{response}</span></div>", unsafe_allow_html=True)



def home():
    import streamlit as st

    with st.expander("Enter Your API Credentials"):
        st.session_state.gemini = st.text_input("Enter Gemini API key to Proceed",type="password")
        st.session_state.Huggingface = st.text_input("Enter Hugging face API key to Proceed",type="password")
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



if selected2 == "Home":
    home()

if st.session_state.gemini and st.session_state.Huggingface:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.Huggingface
    genai.configure(api_key=st.session_state.gemini)
    if  selected2 == "PDF":
        chat_pdf()
    elif selected2 == "Web":
        hyperlink()
    elif selected2 =="CSV":
        chat_csv()
    elif selected2 == "JSON":
        chat_json()
    elif selected2 == "DataBase":
        Chat_db()
else:
    st.warning("Please Provide API keys to procees",icon="⚠️")
