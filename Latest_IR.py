from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def most_similar(doc_id,similarity_matrix,matrix, documents):
    
    top2 = []
    if matrix=='Cosine Similarity':
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix==doc_id:
            continue
    #print(ix, documents[ix])
        if len(top2) == 2:
            return top2
        top2.append(documents[ix])
    return top2
    #return documents[ix]
    #print(f'{matrix} : {similarity_matrix[doc_id][ix]}')
    
def Doc2Vec_Similarity(q, document):
    
    paragraphs = document.split("\n")
    if len(paragraphs) == 1:
        return paragraphs[0]
    paragraphs.append(q)

    tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(paragraphs)]
    #can increase the vector size here
    model_d2v = Doc2Vec(vector_size=100,alpha=0.025, min_count=1)

    model_d2v.build_vocab(tagged_data)

    for epoch in range(10):
        model_d2v.train(tagged_data,
                    total_examples=model_d2v.corpus_count,
                    epochs=model_d2v.epochs)
        
    document_embeddings=np.zeros((len(paragraphs),100))
    n = len(paragraphs)

    for i in range(len(document_embeddings)):
        document_embeddings[i]=model_d2v.docvecs[i]
        
        
    pairwise_similarities=cosine_similarity(document_embeddings)
    pairwise_differences=euclidean_distances(document_embeddings)
    #top2_docs = most_similar(n-1,pairwise_similarities,'Cosine Similarity', paragraphs)
    return most_similar(n-1,pairwise_similarities,'Cosine Similarity', paragraphs)[0]
    #most_similar(n-1,pairwise_differences,'Euclidean Distance', paragraphs)