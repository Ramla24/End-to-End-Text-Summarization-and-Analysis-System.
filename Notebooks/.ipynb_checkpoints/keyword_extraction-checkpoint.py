#!/usr/bin/env python
# coding: utf-8

# In[42]:


import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[44]:


# Download necessary NLTK data



# In[70]:


# Preprocess text: remove special characters, convert to lowercase, remove stopwords
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]

    # Remove words with less than 3 letters
    words = [word for word in words if len(word) >= 3]
    
    return ' '.join(words)


# In[72]:


# Function to extract top n keywords from sorted vector
def extract_topn_from_vector(keywords, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = keywords[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(fname)
    results = {feature_vals[idx]: score_vals[idx] for idx in range(len(feature_vals))}
    return results


# In[74]:


# Function to sort a sparse matrix
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


# In[104]:


# Function to process and extract keywords from a paragraph
def process_paragraph(text, topn=10):
    # Preprocess the input text
    processed_text = preprocess_text(text)
    
    # Initialize CountVectorizer with appropriate parameters for single document
    cv = CountVectorizer(max_df=1.0, min_df=1, max_features=500, ngram_range=(1, 1))
    
    # Fit and transform the text into word count vector
    word_count_vector = cv.fit_transform([processed_text])
    
    # Initialize TF-IDF Transformer
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer = tfidf_transformer.fit(word_count_vector)
    
    # Get feature names (keywords)
    keywords = cv.get_feature_names_out()
    
    # Generate TF-IDF vector for the input text
    tf_idf_vector = tfidf_transformer.transform(cv.transform([processed_text]))
    
    # Sort the TF-IDF vector by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    
    # Extract top n keywords
    top_keywords = extract_topn_from_vector(keywords, sorted_items, topn=topn)
    
    return top_keywords


# In[106]:


paragraph ="It’s not just the specifications that are changing, either. Much has been made of permutations to Google’s algorithms, which are beginning to favor better written, more authoritative content (and making work for the growing content strategy industry). Google’s bots are now charged with asking questions like, “Was the article edited well, or does it appear sloppy or hastily produced?” and “Does this article provide a complete or comprehensive description of the topic?,” the sorts of questions one might expect to be posed by an earnest college professor."


# In[108]:


# # Process the paragraph and extract keywords
# keywords = process_paragraph(paragraph, topn=10)


# In[110]:


# Display the extracted keywords
#print("Extracted Keywords: ", keywords)


# In[112]:


# # Display the extracted keywords line by line
# print("Extracted Keywords:")
# for keyword, score in keywords.items():
#     print(f"{keyword}: {score}")


# In[ ]:




