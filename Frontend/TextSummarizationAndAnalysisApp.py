#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from keyword_extraction import process_paragraph
from text_summarization import generate_abstractive_summary,generate_extractive_summary
from topic_modeling import topicModeling
import streamlit as st
from st_tabs import TabBar
import time


# In[3]:


abstractiveModelFilePath = "../Models/Abstractive Model/peftabstractivemodel"
abstractiveTokenizerFilePath =  "../Models/Abstractive Model/abstractivetokenizer"

extractiveModelFilePath= "../Models/Extractive Model/extractivemodel"
extractiveTokenizerFilePath= "../Models/Extractive Model/extractivetokenizer"


# In[5]:


import re
from bs4 import BeautifulSoup

# Function to clean text with the specific cleaning steps you mentioned
def clean_text(text):
    # 1. HTML Removal
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Special Character and Digit Removal
    text = re.sub(r"[^a-zA-Z0-9.!$?\” \- \“'\s]", '', text)


    # 4. Whitespace Removal
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# In[7]:


@st.cache_resource
def load_abstractive_model():
    return AutoModelForSeq2SeqLM.from_pretrained(abstractiveModelFilePath)

@st.cache_resource
def load_abstractive_tokenizer():
    return AutoTokenizer.from_pretrained(abstractiveTokenizerFilePath)

@st.cache_resource
def load_extractive_model():
    return AutoModelForSeq2SeqLM.from_pretrained(extractiveModelFilePath)

@st.cache_resource
def load_extractive_tokenizer():
    return AutoTokenizer.from_pretrained(extractiveTokenizerFilePath)

# Call the caching functions to load the models/tokenizers
abstractiveModel = load_abstractive_model()
abstractiveTokenizer = load_abstractive_tokenizer()
extractiveModel = load_extractive_model()
extractiveTokenizer = load_extractive_tokenizer()



# In[11]:


# Function for sentiment analysis (placeholder, implement your method)
def analyze_sentiment(text):
    # Your sentiment analysis logic here
    return "Positive"  # Example sentiment

def get_theme(text):
    # Your sentiment analysis logic here
    return "Technology"  # Example sentiment

st.markdown("""
    <style>
    .main .block-container {
       
        max-width: 100%;
    }
    .stTextArea {
        width: 100% !important;
    }
    .stButton {
        width: 100% !important;
    }
    .stTabs [data-baseweb="tab"] {
		height: 50px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
		gap: 2px;
        font-size: 20px;
    }
    .stWrite{
            font-size: 20px}

    </style>
    """, unsafe_allow_html=True)

# Title and subtitle
# Centered Title
if "summary" not in st.session_state:
    st.session_state["summary"] = ""

if "keywords" not in st.session_state:
    st.session_state["keywords"] = []

if "theme" not in st.session_state:
    st.session_state["theme"] = []
st.markdown("<h1 style='text-align: center;'>Text Summarization and Analysis App</h1>", unsafe_allow_html=True)


col1, col2 = st.columns([1, 1])

component1 = TabBar(tabs=["Abstractive", "Extractive"], default=0, color="white", activeColor="red", fontSize="14px")

# Handle your TabBar use cases here
if(component1 == 0):
    # Set the tokenizer and model for Abstractive summarization
    
    tokenizer = abstractiveTokenizer
    model = abstractiveModel
else:
    # Set the tokenizer and model for Extractive summarization
    
    tokenizer = extractiveTokenizer
    model = extractiveModel


# Create a single row for the text input and slider
input_col1,ncol,slider_name,short,slider,long = st.columns([4,10, 2, 1,3,1])  # Adjust the ratio for the input and slider


        

with slider_name:
        # Add the slider directly without nesting
        st.write("Summary Length:")

with slider:
        # Create a container for the slider and the labels
        # Create the slider without numbers (just the slider itself)
     summary_size = st.slider(
    " ", min_value=1, max_value=3, value=2, step=1, format=" ", label_visibility="collapsed"
    )

# Create two columns that take up the full width of the screen
with short:
    st.write("Short")
with long:
    st.write("Long")



maincol1, maincol2 = st.columns([1, 1])
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""


    
   


#upclm,procclmn = st.columns([1,1])

with maincol1:
    input_text = st.text_area("Input text:",value=st.session_state['input_text'], height=400)
    upbtn,empt,procbtn = st.columns([3,2,1])

    with upbtn:
         # File upload option below the slider
       uploaded_file = st.file_uploader(" ", type=["txt"], label_visibility="collapsed")
       if uploaded_file is not None:
         st.session_state['input_text'] = uploaded_file.read().decode("utf-8")
         
    with empt:
       if st.button("Upload File"):
            if uploaded_file is not None:
                # st.session_state['input_text'] = uploaded_file.read().decode("utf-8")  # This is redundant since it's already updated
                input_text = st.session_state['input_text']
                success_message = st.empty()
                success_message.success("File uploaded successfully!")

            # Wait for a few seconds
                time.sleep(3)  # Adjust the delay as needed (3 seconds in this case)

           
                success_message.empty()

         
       
       

   
with procbtn:
        if input_text:
           st.session_state["keywords"] = process_paragraph(input_text,topn=6)
           st.session_state["theme"] = get_theme(input_text)
          
        
        
        if st.button("Process Text"):
            if input_text:
        # Clean and process the input text
               cleantext = clean_text(input_text)

               if(component1 == 0):
                summary_text = generate_abstractive_summary(cleantext, model, tokenizer, length_type=summary_size)
               else:
                summary_text = generate_extractive_summary(cleantext, model, tokenizer, length_type=summary_size)



        # Generate summary
               st.session_state["summary"] = summary_text
               
        

        # Perform sentiment analysis
               sentiment = analyze_sentiment(summary_text)

        # Perform topic modeling
               topics = topicModeling(summary_text)
               
               st.session_state["sentiment"] = sentiment.capitalize()
               st.session_state["topics"] = topics

              


            else:
                st.error("Please enter some text or upload a text file.")


analysiscol1,  analysiscol2 = st.columns([1,1])
with maincol1:
     head,word = st.columns([1,5])
     with head:
          st.subheader("Keywords: ")
          st.subheader(" ")
          st.subheader("Theme: ")
     with word:
          # Create a container for the styled keywords
        keyword_container = st.container()

        # Create a horizontal row of styled keywords
        keyword_row = '<div style="display: flex; flex-wrap: wrap; justify-content: flex-start; gap: 10px;">'
        
        for keyword in st.session_state["keywords"]:
            keyword_row += f'''
            <span style="border: 1px solid #B0BEC5; 
                        border-radius: 20px; 
                        padding: 10px 15px; 
                        margin: 14px; 
                        font-size: 18px; 
                        color: #37474F; 
                        background-color: #E0F7FA; 
                        cursor: pointer;
                        ">
                {keyword.capitalize()}
            </span>
            '''
            keyword_row += '</div>'

        # Render the keyword row in the container
        keyword_container.markdown(keyword_row, unsafe_allow_html=True)
        
        if input_text:
            st.write(" ")
            st.write(" ")
            
            
            
            st.markdown(f"""#### {st.session_state['theme']}""")
          


    
    


with maincol2:
                  summary_text_area = st.text_area("Summary:", value=st.session_state["summary"], height=400, key="summary")
                  if "sentiment" in st.session_state and "topics" in st.session_state:
                         sentiment = st.session_state["sentiment"]
                         st.subheader("Sentiment Analysis:")
                         if sentiment == "Positive":
                              st.markdown('<div style="background-color: green; padding: 15px; border-radius: 5px; display: inline-block; font-size: 18px;">'
                                           f'<strong>{sentiment}</strong>'
                                           '</div>',
                                           unsafe_allow_html=True)
                         else:
                            st.write(sentiment) 

                         st.subheader("Topic Modeling:")
                         for i, topic in enumerate(topics):
                               spaced_topic = topic.title().replace(" ", "&nbsp;")  # Replace spaces with non-breaking spaces
                               st.markdown(f"##### **Topic {i+1}:** {spaced_topic}", unsafe_allow_html=True)

                
       

# In[27]:





# In[ ]:




