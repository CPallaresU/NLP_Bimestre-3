import numpy as np
import pandas as pd

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def one_hot_encoding (corpus):
    
    words_set = {}
    words_set = set(words_set)
    
    for k in corpus:
        
        words = set(k.split(" "))
        
        for i in words :
            
            words_set.add(i)
        
    
    one_h_encod = np.zeros((len(corpus),len(words_set)))
    
    tf_ = np.zeros((len(corpus),len(words_set)))
    
    words_list = list(words_set)
    
    
    for q in range(0,len(corpus)):
        
        
        words_  = list(corpus[q].split(" "))
        
        for h in words_:
                    
            one_h_encod [q][words_list.index(h)] = 1
            tf_[q][words_list.index(h)] = tf_[q][words_list.index(h)] + 1
    
    return pd.DataFrame(one_h_encod,columns = words_list) , pd.DataFrame(tf_ , columns = words_list)





corpus = np.array(['que dia es hoy', 'martes el dia de hoy es martes', 'martes muchas gracias'])




####### ONE HOT ENCODING y TF #######

df_ohe, df_tf = one_hot_encoding(corpus) #ONE HOT ENCODING y TF

####### ONE HOT ENCODING y TF #######





####### IDF #######

idf = np.log10(len(corpus) / df_ohe.sum(axis= 0 )) 

df_idf = pd.DataFrame(idf).T

####### IDF #######







####### TF_IDF #######

tf_idf = df_tf.copy()

cont = 0

for j in df_tf.columns:
    
    tf_idf[j] = df_tf[j]*idf[cont]
    cont = cont + 1 

####### TF_IDF #######





####### COSINE SIMILARITY #######

cosine_m = np.zeros((len(corpus),len(corpus)))

for t in range(len(corpus)):
    
    for w in range(len(corpus)):
    
       a = np.array(tf_idf.iloc[t])
       b = np.array(tf_idf.iloc[w])
       
       cosine_m[t][w] = cosine_similarity(a,b)

####### COSINE SIMILARITY #######


print("CORPUS")
print(corpus)

print(" ")
print(" ")

print("ONE HOT ENCODING")
print(df_ohe)

print(" ")
print(" ")

print("TF")
print(df_tf)

print(" ")
print(" ")

print("IDF")
print(df_idf)

print(" ")
print(" ")

print("TF-IDF")
print(tf_idf)

print(" ")
print(" ")

print("COSINE SIMILARITY")
print(cosine_m)





