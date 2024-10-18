from tqdm import tqdm
from utils import ProcessJudgments,load_pickle, store_as_pickle

from sklearn.metrics.pairwise import cosine_similarity
def store_ranklist(ranklist, filename):
    lines=[]
    for current_case in list(ranklist.keys())[10:]:
        docs = list(ranklist[current_case].keys())
        scores = list(ranklist[current_case].values())
        for doc in docs:
            lines.append(current_case+" Q0 "+doc[0:-4]+" "+str(docs.index(doc)+1)+' '+str(scores[docs.index(doc)])+" "+'doc_level'+'\n')
    file = open('../data/results/'+filename,'w')
    file.writelines(lines)
    file.close()
def transform_queries(query_file_path, vectorizer_filename):
    pickle_path = '../data/pickled_files'
    tfidf_vectorizer = load_pickle(pickle_path, vectorizer_filename)

    query_vectors = {}

    with open(query_file_path, 'r') as f:
        queries = f.readlines()
    i = 1
    for query in tqdm(queries, desc='Vectorizing AILA Queries'):
        cleaned_query = ProcessJudgments.replace_laws(query)
        cleaned_query = ProcessJudgments.process_paragraph(cleaned_query)
        query_vectors['AILA_Q'+ str(i)] = (query,tfidf_vectorizer.transform([cleaned_query]))
        i += 1

    return query_vectors
def score_doc_level(q_vectors, aila_judgment_data):
    query_scores = {}
    for q_id, q_vector in tqdm(q_vectors.items(), desc=f'Computing similarity at Doc level'):
        # iterate over prior docs
        scores = {}
        for doc in aila_judgment_data.keys():
            sim = cosine_similarity(q_vector[1], aila_judgment_data[doc])[0][0]
            scores[doc] = sim
        scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True)) 
        query_scores[q_id] = scores

    return query_scores

def main():
    vectorizer_name = 'tfidf_vectorizer_fitted_to_aila'
    query_file_path = '../data/raw/query_doc.txt'
    pickle_path = '../data/pickled_files'
    aila_judgment_doc_vectors = load_pickle(pickle_path, 'aila_judgment_doc_vectors')
    query_vectors = transform_queries(query_file_path,vectorizer_name)
    


    s1 = score_doc_level(query_vectors, aila_judgment_data =aila_judgment_doc_vectors)
    f1 = 'doc_level'

    store_ranklist(s1, f1)
    store_as_pickle(pickle_path, f1,s1)
if __name__=='__main__':

    main()
