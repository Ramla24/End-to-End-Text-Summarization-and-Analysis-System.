#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.special import softmax


# In[5]:


def sentiment(text, tokenizer,model):
    encoded_txt = tokenizer(text, return_tensors = 'pt')
    
    output = model(**encoded_txt)



    scores = output[0][0].detach().numpy()
    scores = softmax(scores)


    scores_dict = {
    'Negative': scores[0],
    'Neutral': scores[1],
    'Positive': scores[2]
     }


    highest_label = max(scores_dict, key=scores_dict.get)



    return highest_label

    


# In[ ]:




