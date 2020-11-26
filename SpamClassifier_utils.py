import pandas as pd
from os import walk
from os.path import join
import matplotlib.pyplot as plt
import nltk 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import numpy as np

#generator function

def email_body_generator(path):
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            filepath = join(root,file_name)
            #extracting email body
            stream = open(filepath, encoding='latin-1')
            is_body = False
            lines = []
            for line in stream:
                if is_body:
                    lines.append(line)
                elif line =='\n':
                    is_body = True
            stream.close()
            email_body = '\n'.join(lines)
            yield file_name, email_body

def df_from_directory(path, classification):
    rows=[]
    row_names=[]

    for file_name, email_body in email_body_generator(path):
        rows.append({'MESSAGE': email_body,'CATEGORY': classification})
        row_names.append(file_name)
    return pd.DataFrame(rows, index=row_names)

def clean_msg(message, stemmer= PorterStemmer(), stop_words=set(stopwords.words('english'))):

    #remove html tags
    soup = BeautifulSoup(message, 'html.parser')
    text = soup.get_text()

    #convert to Lower Case and split words
    words = word_tokenize(text.lower())
    
    filtered_words = []
    for word in words:
        #removes stop words and punctuations
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))

    return filtered_words

