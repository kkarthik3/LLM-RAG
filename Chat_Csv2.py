import os 
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from pandasai import SmartDataframe
import shutil
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Agg") 
st.set_option('deprecation.showPyplotGlobalUse', False)


import textwrap
from IPython.display import display
from IPython.display import Markdown


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


    prompt = "Don't provide python or any form of codes"
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
        st.write(agent.run("Calculate correlations between numerical variables to identify potential relationships. and don't include Nan values"+prompt))
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
        function_agent1()
        function_agent2()
        st.divider()
        df2 = SmartDataframe(df=st.session_state.df, config={"llm": llm,"open_charts":False,"save_charts":True})
        question()
        query2 = st.text_input("Give query to plot",key="3")
        if query2:
            plot(query2)


chat_csv()