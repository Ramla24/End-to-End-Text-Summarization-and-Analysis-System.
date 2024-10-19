#!/usr/bin/env python
# coding: utf-8

# In[13]:


import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
import re

# Initialize stopwords, punctuation set, and lemmatizer
stop = set(stopwords.words('english'))  # If you are using NLTK stopwords
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Function to lemmatize and remove the stopwords
def clean(text):
    stop_words = set(stopwords.words('english'))
    words = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text.lower())  # Tokenize and lowercase
    words = [word for word in words if word.isalnum()]  # Remove punctuation
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemma.lemmatize(word) for word in words if len(word) >3]
    return words

# In[33]:


def topicModeling(text):
    cleantext = clean(text)


    # Create dictionary and corpus
    dictionary = corpora.Dictionary([cleantext])
    corpus = [dictionary.doc2bow(cleantext)]
    # Creating the LDA model
    ldamodel = LdaModel(corpus=corpus, num_topics=3,id2word=dictionary,  passes=50)
    topics = ldamodel.show_topics(num_words=8)
    # for topic in topics:
    #     print(topic)
    topic_distribution = ldamodel.get_document_topics(corpus[0])
    formatted_topics=[]
    
    for topic in topics:
        top=(topic[1])
        
        words = top.split('+')
        
        formatted_list = []
        for component in words:
        # Remove the quotes and split into coefficient and word
             coefficient, word = component.split('*')
             word = word.strip().strip('"')  # Remove the surrounding quotes
             formatted_list.append(f"{word} ")

    # Step 3: Join the formatted list into a single string
        formatted_topics.append(',  '.join(formatted_list))

    return formatted_topics
    


# In[35]:

# In[ ]:




