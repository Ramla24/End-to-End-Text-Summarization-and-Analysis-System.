#!/usr/bin/env python
# coding: utf-8

# In[13]:




# In[6]:


def generate_abstractive_summary(text, model, tokenizer, length_type=2, max_length=1024, overlap=512):
    # Clean and chunk the input text
    cleantext = text
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
           
            max_length=s_max_length,
            do_sample=True,                   # Enable sampling for more abstract, creative output
            top_p=0.5,                        # Lower top-p to encourage less common word choices
            top_k=40,                         # Use top-k for more abstractive generation
            temperature=2.5,                  # Increase temperature for more diverse outputs
            no_repeat_ngram_size=4,           # Increase to 4 to discourage exact phrases
            repetition_penalty=4.0,           # Strong repetition penalty to avoid similar phrasing
            length_penalty=0.5,               # Lower length penalty for shorter, more concise summaries
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generated_summaries.append(summary_text)

    # Combine all generated summaries
    final_summary = " ".join(generated_summaries)

    # Remove redundant sentences only if there is more than one chunk
    if len(text_chunks) > 1:
        final_summary = remove_repetitive_sentences(final_summary)

    return final_summary


# In[7]:


def generate_extractive_summary(text, model, tokenizer, length_type=2, max_length=1024, overlap=512):
    # Clean and chunk the input text
    cleantext = text
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
            num_beams=5,                  # Use beam search to find the best combinations
            max_length=s_max_length,      # Set the max length for the summary
            length_penalty=1.0,          # Encourage longer sentences
            no_repeat_ngram_size=3,       # Prevent repeating phrases
            early_stopping=True,          # Stop early when the best sequences are found
            do_sample=False,              # Disable sampling for deterministic output
            temperature=1.0,              # Keep temperature low for focused output
            top_k=0,                      # Disable top-k sampling
            top_p=1.0                     # Disable nucleus sampling
        )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generated_summaries.append(summary_text)

    # Combine all generated summaries
    final_summary = " ".join(generated_summaries)

    # Remove redundant sentences only if there is more than one chunk
    if len(text_chunks) > 1:
        final_summary = remove_repetitive_sentences(final_summary)

    return final_summary


# In[8]:


def remove_repetitive_sentences(summary):
    sentences = summary.split(". ")  # Split the summary into sentences
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        if sentence.strip() not in seen_sentences:
            unique_sentences.append(sentence.strip())
            seen_sentences.add(sentence.strip())

    return ". ".join(unique_sentences) + "."


# In[9]:


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



# In[10]:


# input_text = """Several of the world’s leading technology companies, including Google, Microsoft, and OpenAI, have joined forces to establish new safety standards for the development and deployment of artificial intelligence (AI). This collaboration comes amid growing concerns about the potential risks posed by advanced AI systems, particularly in critical areas such as healthcare, defense, and finance.

# The companies announced their plans at a global summit on AI governance, held in San Francisco. Their goal is to create a shared framework that ensures AI systems are developed in a way that prioritizes human safety, transparency, and accountability. The framework will include guidelines for testing AI models, ensuring fairness in decision-making processes, and preventing unintended harmful consequences.

# In a joint statement, the tech giants emphasized the need for international cooperation, urging governments and regulatory bodies to work alongside the private sector in shaping the future of AI. "We recognize the tremendous potential of AI, but we must also ensure that it is used responsibly and for the benefit of all," said Sundar Pichai, CEO of Google.

# This move is seen as a proactive step by the tech industry to address ethical and safety concerns before governments enforce stricter regulations. Experts believe that this initiative will help mitigate some of the risks associated with AI while allowing innovation to thrive in a controlled environment.

# However, critics argue that the companies involved may have conflicts of interest, as they stand to profit from AI development. Some have called for more independent oversight and stricter governmental regulation to ensure that safety and ethical standards are maintained.

# The AI safety framework is expected to be finalized by mid-2025 and will be made publicly available for adoption by other companies and institutions globally."""


# # In[80]:


# print(input_text)


# # In[11]:


# extsmall = generate_extractive_summary(input_text,model,tokenizer)


# # In[146]:


# absmall = generate_absractive_summary(input_text,model,tokenizer)


# # In[132]:


# extsmall


# # In[138]:


# extsmall


# # In[144]:


# extsmall


# # In[63]:


# extsmall

