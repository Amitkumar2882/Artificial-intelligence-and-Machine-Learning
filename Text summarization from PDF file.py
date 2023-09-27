#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer
import heapq


# In[2]:


# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Function to summarize text
def summarize_text(text, num_sentences=5):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Remove stopwords and punctuation
    stopwords_list = set(stopwords.words("english"))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords_list]

    # Calculate word frequency
    word_freq = FreqDist(words)

    # Calculate sentence scores based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

    # Get the top 'num_sentences' sentences with the highest scores
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Detokenize the summary sentences
    summary = TreebankWordDetokenizer().detokenize(summary_sentences)

    return summary


# In[3]:


# Function to read and summarize PDF content
def summarize_pdf(pdf_file, num_sentences=5):
    try:
        pdf_text = ""
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_text += page.extract_text()

        summary = summarize_text(pdf_text, num_sentences)
        return summary

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_file_path = 'Operations Management.pdf'
    num_summary_sentences = 5
    summary = summarize_pdf(pdf_file_path, num_summary_sentences)
    
    if summary:
        print("Summary:")
        print(summary)
    else:
        print("Failed to summarize the PDF.")


# In[ ]:




