
from tqdm import tqdm
from utils import ProcessJudgments,load_pickle,store_as_pickle

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def store_ranklist(ranklist, filename):
    lines=[]
    for current_case in list(ranklist.keys())[10:]:
        docs = list(ranklist[current_case].keys())
        scores = list(ranklist[current_case].values())
        for doc in docs:
            lines.append(current_case+" Q0 "+doc[0:-4]+" "+str(docs.index(doc)+1)+' '+str(scores[docs.index(doc)])+" "+'para_level'+'\n')
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
def score_org_docs(q_vectors, aila_judgment_data, k_for_P):
    query_scores = {}
    for q_id, q_vector in tqdm(q_vectors.items(), desc=f'Computing similarity using constituent paragraphs with k = {k_for_P}'):
        # iterate over all prior docs
        scores = {}
        for doc in aila_judgment_data.keys():
            sim_values = cosine_similarity(q_vector[1], aila_judgment_data[doc])
            sim_values = sim_values.reshape(-1)
            
            sorted_index_array = np.argsort(sim_values)
            sorted_sim_values = sim_values[sorted_index_array]

            sim = np.mean(sorted_sim_values[-k_for_P : ])
            scores[doc] = sim
        scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True)) 
        query_scores[q_id] = scores
    return query_scores
def score_citation_anchor_text(q_vectors, high_confidence_mapping_aila, citation_anchor_text, k_for_CAT, aila_prior_data):
    query_scores = {}
    
    for q_id, q_vector in tqdm(q_vectors.items(), desc=f'Computing similarity using CAT with k = {k_for_CAT}'):
    # iterate over aila docs
        scores = {}
        for doc in aila_prior_data.keys():
            # if doc is present in citation graph, then get the score
            if doc in high_confidence_mapping_aila.keys() and high_confidence_mapping_aila[doc][0] in citation_anchor_text.keys():
                reqd_ik_doc = high_confidence_mapping_aila[doc][0]
                sim_values = cosine_similarity(q_vector[1], citation_anchor_text[reqd_ik_doc])

                sim_values = sim_values.reshape(-1)

                if sim_values.shape[0] > k_for_CAT:
                    sorted_index_array = np.argsort(sim_values)
                    sorted_msp = sim_values[sorted_index_array]
                    scores[doc] = np.mean(sorted_msp[-k_for_CAT : ])
                elif  sim_values.shape[0] != 0:
                    sorted_index_array = np.argsort(sim_values)
                    sorted_msp = sim_values[sorted_index_array]
                    scores[doc] = np.mean(sorted_msp[-sim_values.shape[0] : ])
                else:
                    # if no paras citing then assign 0
                    scores[doc] = 0
            else:
                # if doc not in citation graph, need not add any score because scores from original docs are in separate function
                scores[doc] = 0
        scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True)) 
        query_scores[q_id] = scores
    return query_scores
def main():
    vectorizer_name = 'tfidf_vectorizer_fitted_to_aila'
    query_file_path = '../data/raw/query_doc.txt'
    pickle_path = '../data/pickled_files'
    ik_pickle_path = '../../IK/data/pickled_files'
    aila_prior_data = load_pickle(pickle_path, 'aila_prior_data')
    citation_anchor_text = load_pickle(pickle_path, 'citation_anchor_text')
    high_confidence_mapping_aila = load_pickle(ik_pickle_path, 'high_confidence_mapping_aila')
    query_vectors = transform_queries(query_file_path,vectorizer_name)
    
    values_of_k = list(range(1,11))
    for k_for_P in values_of_k:
        s1 = score_org_docs(query_vectors, aila_judgment_data =aila_prior_data,k_for_P=k_for_P)
        f1 = 'P_'+str(k_for_P)
        store_ranklist(s1, f1)
        store_as_pickle(pickle_path, f1, s1)

    for k_for_CAT in values_of_k:
        s2 = score_citation_anchor_text(query_vectors,high_confidence_mapping_aila=high_confidence_mapping_aila, 
                                              citation_anchor_text=citation_anchor_text,
                                              k_for_CAT = k_for_CAT, 
                                              aila_prior_data= aila_prior_data)
        f2 = 'CAT_'+str(k_for_CAT)
        store_ranklist(s2, f2)
        store_as_pickle(pickle_path, f2, s2)
   

if __name__=='__main__':
    main()
