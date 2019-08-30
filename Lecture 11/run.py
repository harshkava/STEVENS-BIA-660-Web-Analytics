"""
Using Latent Dirichlet Allocation for topic modeling. 
"""

from sklearn.feature_extraction.text import CountVectorizer
import lda
import numpy as np

topic_num=20

#tokenization
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
                                
#read the dataset                
docs=open('news.txt').readlines()

#transform the docs into a count matrix
matrix = tf_vectorizer.fit_transform(docs)

#get the vocabulary
vocab=tf_vectorizer.get_feature_names()

#initialize the LDA model
model = lda.LDA(n_topics=topic_num, n_iter=500)

#fit the model to the dataset
model.fit(matrix)

#write the top terms for each topic
top_words_num=20
topic_mixes= model.topic_word_

fw=open('top_terms_per_topic.txt','w')
for i in range(topic_num):#for each topic
    top_indexes=np.argsort(topic_mixes[i])[::-1][:top_words_num]                              
    my_top=''
    for ind in top_indexes:my_top+=vocab[ind]+' ' 
    fw.write('TOPIC: '+str(i)+' --> '+str(my_top)+'\n')
fw.close()


#write the top topics for each doc
top_topics_num=3
doc_mixes= model.doc_topic_
fw=open('topic_mixture_per_doc.txt','w')
for i in range(len(doc_mixes)):#for each doc
    top_indexes=np.argsort(doc_mixes[i])[::-1][:top_topics_num]     
    my_top=''
    for ind in top_indexes:my_top+=' '+str(ind)+':'+str(round(doc_mixes[i][ind],2))
    fw.write('DOC: '+str(i)+' --> '+str(my_top)+'\n')
fw.close()






