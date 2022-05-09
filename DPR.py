from sklearn.metrics import precision_score,recall_score,f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer, util
import numpy as np
passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
import torch


def get_monstor_passage_split(monstor_text, monstor_name):
    if len(monstor_text) == 0:
        return []
    count = 0
    data_dpr = []
    for docs in monstor_text.split("\n"):
        id = monstor_name + str(count)
        content = docs
        new_pass = id + " [SEP] " + content
        data_dpr.append((id, new_pass))
        count += 1
    return data_dpr

def get_most_similar(question, monstor_text, corpse_page, monstor_name):
    monstor_passage_split_data = get_monstor_passage_split(monstor_text,monstor_name)
    corpse_passage_split_data = get_monstor_passage_split(corpse_page,"corpse")
    monstor_passages = [x[1] for x in monstor_passage_split_data]
    corpse_passages = [x[1] for x in corpse_passage_split_data]
    passages = monstor_passages + corpse_passages
    if len(passages) == 0:
        return "",""

    passage_embeddings = passage_encoder.encode(passages)
    query_embedding = query_encoder.encode(question)
    scores = util.dot_score(query_embedding, passage_embeddings)

    ind = torch.argmax(scores).item()
    docid, text_1  = passages[ind].split("[SEP]")
    return  docid,text_1