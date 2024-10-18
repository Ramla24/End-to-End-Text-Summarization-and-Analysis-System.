#!/usr/bin/env python
# coding: utf-8

# In[42]:


import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tag import pos_tag


# In[44]:

nltk.data.path.append('C:\\Users\\ASUS\\AppData\\Roaming\\nltk_data')
# Download necessary NLTK data
nltk.download('averaged_perceptron_tagger')



# In[70]:

useful_three_letter_words = {
    'act', 'aim', 'arm', 'art', 'ant', 'box', 'cap', 'car', 'cat', 'day', 
    'dog', 'end', 'eye', 'ear', 'fat', 'gas', 'job', 'law', 'map', 'net', 
    'sky', 'bit', 'set', 'log', 'key', 'pin', 'fan', 'fit', 
    'tag', 'zip', 'vet', 'bat', 'man', 'tap','ai'
}


# Preprocess text: remove special characters, convert to lowercase, remove stopwords
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    
    text = re.sub(r'\d+', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Remove extra spaces
    
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    modal_verbs = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}
    stop_words.update(modal_verbs)
    words = nltk.word_tokenize(text)

    words = [word for word in words if word not in stop_words]

    # Remove words with less than 3 letters
    words = [word for word in words if len(word) > 3 or word in useful_three_letter_words]
    

    return ' '.join(words)

    
   


# In[72]:
# preprocess_text("""Several of the world’s leading technology companies, including Google, Microsoft, and OpenAI, have joined forces to establish new safety standards for the development and deployment of artificial intelligence (AI). This collaboration comes amid growing concerns about the potential risks posed by advanced AI systems, particularly in critical areas such as healthcare, defense, and finance.

# The companies announced their plans at a global summit on AI governance, held in San Francisco. Their goal is to create a shared framework that ensures AI systems are developed in a way that prioritizes human safety, transparency, and accountability. The framework will include guidelines for testing AI models, ensuring fairness in decision-making processes, and preventing unintended harmful consequences.

# In a joint statement, the tech giants emphasized the need for international cooperation, urging governments and regulatory bodies to work alongside the private sector in shaping the future of AI. "We recognize the tremendous potential of AI, but we must also ensure that it is used responsibly and for the benefit of all," said Sundar Pichai, CEO of Google.

# This move is seen as a proactive step by the tech industry to address ethical and safety concerns before governments enforce stricter regulations. Experts believe that this initiative will help mitigate some of the risks associated with AI while allowing innovation to thrive in a controlled environment.

# However, critics argue that the companies involved may have conflicts of interest, as they stand to profit from AI development. Some have called for more independent oversight and stricter governmental regulation to ensure that safety and ethical standards are maintained.

# The AI safety framework is expected to be finalized by mid-2025 and will be made publicly available for adoption by other companies and institutions globally
# """)


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




