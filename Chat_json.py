import os 
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from pandasai import SmartDataframe

import textwrap
from IPython.display import display
from IPython.display import Markdown

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


chat_json()