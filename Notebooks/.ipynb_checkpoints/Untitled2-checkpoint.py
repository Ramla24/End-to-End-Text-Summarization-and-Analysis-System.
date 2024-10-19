#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from keyword_extraction import process_paragraph


import streamlit as st
import os


# In[3]:


modelFilePath = "../Models/Text Summarization Model/abstractivemodel"
tokenizerFilePath =  "C:\\Users\\ASUS\\Documents\\y3s1\\IRWA\\Project\\IRWA_Project\\Models\\Text Summarization Model\\abstractivetokenizer"


print(os.getcwd())


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



tokenizer = AutoTokenizer.from_pretrained(tokenizerFilePath)
model = AutoModelForSeq2SeqLM.from_pretrained(modelFilePath)


# In[9]:


def generate_summary_for_long_text(text, model, tokenizer, length_type=1, max_length=1024, overlap=512):
    # Clean and chunk the input text
    cleantext = clean_text(text)
    text_chunks = chunk_text_with_overlap(cleantext, tokenizer, max_length, overlap)

    generated_summaries = []
    
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"]
        input_length = inputs['input_ids'].shape[-1]

        if length_type == 1:
            s_max_length = int(input_length * 0.2)
        elif length_type == 2:
            s_max_length = int(input_length * 0.4)
        elif length_type == 3:
            s_max_length = int(input_length * 0.6)

        s_max_length = min(s_max_length, 1024)

        summary_ids = model.generate(
            input_ids,
            num_beams=5,
            max_length=s_max_length,
            no_repeat_ngram_size=3,  # Prevent repeated n-grams
            length_penalty=2.0,
            temperature=0.9,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id
        )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generated_summaries.append(summary_text)

    # Combine all generated summaries
    final_summary = " ".join(generated_summaries)

    # Remove redundant sentences only if there is more than one chunk
    if len(text_chunks) > 1:
        final_summary = remove_repetitive_sentences(final_summary)

    return final_summary


# In[11]:


def remove_repetitive_sentences(summary):
    sentences = summary.split(". ")  # Split the summary into sentences
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        if sentence.strip() not in seen_sentences:
            unique_sentences.append(sentence.strip())
            seen_sentences.add(sentence.strip())

    return ". ".join(unique_sentences) + "."


# In[17]:


# Define the max token length for the model and overlap size
max_token_length = 1024  # Modify based on your model's token limit
overlap_size = 256  # Number of tokens to overlap

# Function to chunk text with overlap
def chunk_text_with_overlap(text, tokenizer, max_length, overlap):
    tokens = tokenizer.tokenize(text)
    chunks = []

    # Create the first chunk
    first_chunk = tokens[:max_length]  # Get the first chunk
    chunk_text = tokenizer.convert_tokens_to_string(first_chunk)
    chunks.append(chunk_text)

    # If the first chunk is less than max_length, we don't need more chunks
    if len(first_chunk) < max_length:
        return chunks

    # Continue chunking with overlap for the rest of the text
    for i in range(max_length - overlap, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)

    return chunks



# In[19]:


#input_text = """Text summarization is one of the many applications of natural language processing. There are two types of summarization models available. One is the extractive text summarization and the other one is the abstractive text summarization. In extractive text summarization, after the stop words are removed from the text, the frequency of occurrence of each word token is calculated. Each token is assigned a rank or weight based on frequency of occurrence, higher the frequency, greater the weight. Next, each sentence is assigned a weight by summing over the individual weights of each token present in it. The sentences are ranked per the weight and the k topmost ranking sentences are presented as the summary. While a model like this does not need any training, but this is rule based. In my personal experience, longer sentences are often selected and many a times it fails to capture the context of the text. Therefore, this article is about abstractive text summarization, which is a supervised learning model built using a transformer."""


# In[21]:

#ext = generate_summary_for_long_text(input_text, model, tokenizer, length_type=2)


# In[23]:

#ext



# In[25]:




# Function for sentiment analysis (placeholder, implement your method)
def analyze_sentiment(text):
    # Your sentiment analysis logic here
    return "Positive"  # Example sentiment

# Function for topic modeling (placeholder, implement your method)
def perform_topic_modeling(text):
    # Your topic modeling logic here
    return ["Topic 1", "Topic 2"]

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
    </style>
    """, unsafe_allow_html=True)

# Title and subtitle
# Centered Title
if "summary" not in st.session_state:
    st.session_state["summary"] = ""

if "keywords" not in st.session_state:
    st.session_state["keywords"] = []
st.markdown("<h1 style='text-align: center;'>Text Summarization and Analysis App</h1>", unsafe_allow_html=True)


col1, col2 = st.columns([1, 1])

# Create a single row for the text input and slider
input_col1, slider_name,short,slider,long = st.columns([10, 2, 1,3,1])  # Adjust the ratio for the input and slider

with input_col1:
        st.write("")
        # Input text area on the left
        

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

with maincol1:
    input_text = st.text_area("Enter your text:", height=400)
   
if input_text:
    st.session_state["keywords"] = process_paragraph(input_text,topn=10)

#upclm,procclmn = st.columns([1,1])

with maincol1:
    upbtn,empt,procbtn = st.columns([3,2,1])

    with upbtn:
         # File upload option below the slider
       uploaded_file = st.file_uploader("", type=["txt"], label_visibility="collapsed")
       if uploaded_file is not None:
         input_text = uploaded_file.read().decode("utf-8")
   
    with procbtn:
        
        
        if st.button("Process Text"):
            if input_text:
        # Clean and process the input text
               cleantext = clean_text(input_text)

        # Generate summary
               summary_text = generate_summary_for_long_text(input_text, model, tokenizer, length_type=summary_size)
               
               st.session_state["summary"] = summary_text
               
        

        # Perform sentiment analysis
               sentiment = analyze_sentiment(summary_text)

        # Perform topic modeling
               topics = perform_topic_modeling(summary_text)


               st.session_state["sentiment"] = sentiment
               st.session_state["topics"] = topics


              
                    
               
               
        # Display keywords, sentiment, and topic modeling results
            #    with maincol1:
            #       st.write(", ".join(keywords))  # Display keywords under input text box

               
                  #sentiment_placeholder.write(sentiment)  # Display sentiment analysis
                  #topic_placeholder.write(", ".join(topics))  # Display topic modeling results


            else:
                st.error("Please enter some text or upload a text file.")


analysiscol1,  analysiscol2 = st.columns([1,1])
with maincol1:
     head,word = st.columns([1,2])
     with head:
          st.subheader("Keywords:")
     with word:
          st.write(", ".join(st.session_state["keywords"]))


with maincol2:
                  summary_text_area = st.text_area("Summary:", value=st.session_state["summary"], height=400, key="summary")
                  if "sentiment" in st.session_state and "topics" in st.session_state:
                         st.subheader("Sentiment Analysis:")
                         st.write(st.session_state["sentiment"])  # Display the sentiment result

                         st.subheader("Topic Modeling:")
                         st.write(", ".join(st.session_state["topics"]))

                
       

# In[27]:





# In[ ]:




