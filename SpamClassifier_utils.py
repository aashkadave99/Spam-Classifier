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

def sparse_matrix(df, index_words, labels): 
    
    #df: dataframe with words in the columns with a document id as an index (X_train or X_test) ; indexed_words: index of words ordered by word_id
    #returns sparse matrix

    nr_rows = df.shape[0]
    nr_cols = df.shape[1]
    word_set = set(index_words)
    dictList = []

    for i in range(nr_rows):
        for j in range(nr_cols):
            word = df.iat[i,j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = index_words.get_loc(word)
                cat = labels.at[doc_id]
                item = {'LABEL':cat, 'DOC_ID':doc_id, 'OCCURENCE':1, 'WORD_ID':word_id}
                dictList.append(item)

    return pd.DataFrame(dictList) 

def make_full_matrix(sparse_matrix, nr_words, doc_idx=0, word_idx=1, cat_idx=2, freq_idx =3,):
    '''
    Form a full matrix from a sparse matrix. Return panda dataframe
    sparse_matrix -- numpy array
    nr_words -- size of vocabulary, total number of tokens
    doc_idx -- position of document id in sparse matrix default: 1st col
    word -- position of word id in sparse matrix default: 2nd col
    cat_idx -- position of label in sparse matrix default: 3rd col
    freq_idx -- position of occurence of word in sparse matrix default: 4th col
    '''
    col_names = ['DOC_IDS'] + ['CATEGORY'] + list(range(0,nr_words))
    doc_id_names = np.unique(sparse_matrix[:,0])
    full_matrix = pd.DataFrame(index= doc_id_names, columns= col_names).fillna(value=0)
    
    for i in range(sparse_matrix.shape[0]):
        
        doc_nr = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        label = sparse_matrix[i][cat_idx]
        occurence = sparse_matrix[i][freq_idx]
        
        full_matrix.at[doc_nr, 'DOC_ID']= doc_nr
        full_matrix.at[doc_nr, 'CATEGORY']= label
        full_matrix.at[doc_nr, word_id]= occurence
        
    full_matrix.set_index('DOC_ID', inplace=True)
    
    return full_matrix