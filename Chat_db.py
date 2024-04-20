import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import io
import csv
import pandas as pd

st.set_page_config(page_title="Chat with Your Databases", layout="wide")
st.title("Chat with MySQL")


llm = ChatGoogleGenerativeAI(model="models/gemini-pro", google_api_key=os.getenv("google_api_key"),
                                temperature=1,convert_system_message_to_human=True)

def Chat_db():

    def init_database(user: str, password: str, host: str, port: str, database: str) :
        db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        db = SQLDatabase.from_uri(db_uri)
        return db

    def get_query(db):
        template = """You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would answer the user's question.:
        {schema}

        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

        For example:
        Question: which 3 artists have the most tracks?
        SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
        Question: Name 10 artists
        SQL Query: SELECT Name FROM Artist LIMIT 10;

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
            return db.get_table_info()

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
                st.markdown(f"<div style='background-color:#183c2c; padding: 10px; border-radius: 5px;'><span style='font-size:20px;'>{sql_response}</span></div>", unsafe_allow_html=True)
            with st.spinner("The Data is loading ...."):
                data = st.session_state.db.run(f"{sql_response}")
                st.dataframe(str_csv(data))
            with st.spinner("Your Query is Performing"):
                response  = answering(Query, st.session_state.db)
                st.markdown(f"<div style='background-color:#183c2c; padding: 10px; border-radius: 5px;'><span style='font-size:20px;'>{response}</span></div>", unsafe_allow_html=True)

Chat_db()