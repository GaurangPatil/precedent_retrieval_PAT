from tqdm import tqdm
from utils import ProcessJudgments,load_pickle,store_as_pickle
import copy

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sp
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
def score_p_union_cat(q_vectors, aila_judgment_data, k_for_P_union_CAT, high_confidence_mapping_aila, citation_anchor_text):
    query_scores = {}
    for q_id, q_vector in tqdm(q_vectors.items(), desc=f'Computing similarity using P U CAT with k = {k_for_P_union_CAT}'):
        # iterate over prior  docs
        scores = {}
        for doc in aila_judgment_data.keys():
            # get vectors from original paragraphs
            P = aila_judgment_data[doc]
            
            final_representation = copy.deepcopy(P)

            # if CAT is available
            if doc in high_confidence_mapping_aila.keys() and high_confidence_mapping_aila[doc][0] in citation_anchor_text.keys():
                reqd_ik_doc = high_confidence_mapping_aila[doc][0]
                CAT = citation_anchor_text[reqd_ik_doc]
                final_representation = sp.vstack((P,CAT))
            sim_values = cosine_similarity(q_vector[1],final_representation)

            
            sim_values = sim_values.reshape(-1)
            
            sorted_index_array = np.argsort(sim_values)
            sorted_sim_values = sim_values[sorted_index_array]

            sim = np.mean(sorted_sim_values[-k_for_P_union_CAT : ])
            scores[doc] = sim
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
    for k_for_P_union_CAT in values_of_k:
        s1 = score_p_union_cat(query_vectors, aila_prior_data, k_for_P_union_CAT, high_confidence_mapping_aila, citation_anchor_text)
        f1 = 'P_U_CAT_'+str(k_for_P_union_CAT)
        store_ranklist(s1, f1)
        store_as_pickle(pickle_path, f1, s1)

if __name__=='__main__':

    main()
