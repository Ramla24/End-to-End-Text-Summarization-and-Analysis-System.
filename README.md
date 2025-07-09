##🧠 End-to-End Text Summarization and Analysis System
An NLP-powered Streamlit web app for real-time news summarization, sentiment analysis, keyword extraction, and topic modeling.

📌 Overview
This project is a comprehensive NLP application that uses state-of-the-art large language models (LLMs) to automate the process of analyzing and summarizing news articles. It helps users grasp the core ideas of lengthy texts quickly and efficiently through:

📝 Abstractive & Extractive Summarization
💬 Sentiment Analysis
🔍 Keyword Extraction
🧵 Topic Modeling

The application is built with a strong focus on usability and performance, making it suitable for journalists, researchers, and information seekers who need real-time insights from large volumes of text.

🎯 Features
🔹 Text Summarization
Abstractive Summarization using fine-tuned BART & T5 models.
Extractive Summarization using fine-tuned BERT.
Custom summary length and type selection.

🔹 Sentiment Analysis
Real-time analysis using TinyLLaMA and VADER.
Supports detection of positive, negative, and neutral tones.

🔹 Topic Modeling
Implemented with LDA (Latent Dirichlet Allocation).
Visualized with word clouds for better interpretability.

🔹 Keyword Extraction
Built using TF-IDF algorithm.
Preprocessed with POS tagging, stopword removal, and verb filtering.

🧰 Tech Stack
Layer	Tools & Technologies
Frontend	- Streamlit
Backend	- Python, Transformers, SpaCy, NLTK
Models - BART, T5, BERT, TinyLLaMA, VADER
Trainin - Google Colab, Jupyter Notebook
Version Control - GitHub
Deploymen - Streamlit Sharing

⚙️ System Architecture
User → Streamlit UI → Backend (Python + Models) → Output:
Summary, Keywords, Sentiment, Topics


🚀 How to Run Locally
Clone the repository:
git clone https://github.com/Ramla24/Text-Summarization-and-Analysis-System.git
cd yText-Summarization-and-Analysis-System

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
cd Frontend
streamlit run app.py

🧪 Model Training
Fine-tuned BART and T5 on the MultiNews dataset using LoRA for efficient training.
Fine-tuned BERT using a custom dataset of news articles for extractive summarization.
TinyLLaMA was fine-tuned using the News Sentiment Analysis dataset.



📈 Evaluation Metrics
Summarization: ROUGE-1, ROUGE-2, ROUGE-L
Sentiment Analysis: Accuracy, Precision, Recall, F1-score
Topic Modeling: Coherence Score, Perplexity Score
Keyword Extraction: Manual and TF-IDF validation



🎯 Use Cases
🗞️ News summarization for journalists and researchers.
📈 Business intelligence from news content.

🔥 Future Enhancements
🌍 Multilingual support
🧠 Integration of advanced models like GPT-4
☁️ Cloud-based deployment (e.g., AWS, Azure)
🔁 Real-time streaming input support
🎯 Advanced customization and fine-grained sentiment analysis

## 📽️ Demo Video

🎥 [Click here to watch the demo](https://drive.google.com/file/d/1y1fKNCHWXeyLKBdbpLz6EnvR6tz7wBKW/view?usp=sharing)  
*This video walks through the app features and live output on real news data.*


## 📄 Final Report

📘 [Download the full report (PDF)](./Report.pdf)  
A comprehensive technical breakdown including architecture, implementation, fine-tuning, evaluations, and future improvements.

📚 References
MultiNews Dataset
News Sentiment Dataset
Hugging Face Transformers
YouTube tutorials for model fine-tuning
Streamlit and NLP libraries documentation



# 🧠 End-to-End Text Summarization and Analysis System

An NLP-powered Streamlit web app for real-time news summarization, sentiment analysis, keyword extraction, and topic modeling.

---

## 📌 Overview

This project is a comprehensive NLP application that uses state-of-the-art large language models (LLMs) to automate the process of analyzing and summarizing news articles. It helps users grasp the core ideas of lengthy texts quickly and efficiently through:

- 📝 **Abstractive & Extractive Summarization**
- 💬 **Sentiment Analysis**
- 🔍 **Keyword Extraction**
- 🧵 **Topic Modeling**

The application is built with a strong focus on usability and performance, making it suitable for journalists, researchers, and information seekers who need real-time insights from large volumes of text.

---

## 🎯 Features

### 🔹 Text Summarization
- Abstractive Summarization using fine-tuned **BART** & **T5** models.
- Extractive Summarization using fine-tuned **BERT**.
- Customizable summary length and type selection.

### 🔹 Sentiment Analysis
- Real-time sentiment detection using **TinyLLaMA** and **VADER**.
- Classifies content as **positive**, **negative**, or **neutral**.

### 🔹 Topic Modeling
- Implemented using **LDA (Latent Dirichlet Allocation)**.
- Visualized with word clouds for better interpretability.

### 🔹 Keyword Extraction
- Built using a custom **TF-IDF** algorithm.
- Enhanced with **POS tagging**, **stopword removal**, and **verb filtering**.

---

## 🧰 Tech Stack

| Layer           | Tools & Technologies                    |
|-----------------|------------------------------------------|
| **Frontend**    | Streamlit                                |
| **Backend**     | Python, Transformers, SpaCy, NLTK        |
| **Models**      | BART, T5, BERT, TinyLLaMA, VADER         |
| **Training**    | Google Colab, Jupyter Notebook           |
| **Versioning**  | GitHub                                   |
| **Deployment**  | Streamlit Sharing                        |

---

## ⚙️ System Architecture
User → Streamlit UI → Backend (Python + Models) → Output:
Summary, Keywords, Sentiment, Topics



---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ramla24/Text-Summarization-and-Analysis-System.git
   cd Text-Summarization-and-Analysis-System

2. **Install dependencies**
3. 

