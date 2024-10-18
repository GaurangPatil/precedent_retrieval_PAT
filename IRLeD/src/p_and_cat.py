
from tqdm import tqdm

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from utils import store_as_pickle, load_pickle

def score_org_paragraphs(irled_judgment_data_prior, k_for_P, irled_query_data):
    query_scores = {}
    # iterate over all query cases
    for query in tqdm(irled_query_data.keys(),desc=f'Computing similarity using constituent paragraphs with k = {k_for_P}'):

        scores = {}
        # iterate over all prior judgments
        for doc in irled_judgment_data_prior.keys():

            sim_values = cosine_similarity(irled_query_data[query], irled_judgment_data_prior[doc])

      
            
            sim_values = sim_values.reshape(-1)
            
            sorted_index_array = np.argsort(sim_values)
            sorted_sim_values = sim_values[sorted_index_array]

            sim = np.mean(sorted_sim_values[-k_for_P : ])
            scores[doc] = sim

        scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True)) 
        query_scores[query] = scores
            
        
    return query_scores
def score_citation_anchor_text( citation_anchor_text, k_for_CAT, irled_query_data, irled_prior_data, high_confidence_mapping_prior):
    query_scores = {}
    # iterate over all query cases
    for query in tqdm(irled_query_data.keys(),desc=f'Computing similarity using CAT with k = {k_for_CAT}'):

        scores = {}
        # iterate over all prior cases
        for doc in irled_prior_data.keys():

            # compute score from CAT only if prior case is mapped with similarity above threshold
            if doc in high_confidence_mapping_prior.keys():

                para_vectors = citation_anchor_text[high_confidence_mapping_prior[doc][0]]
                sim_values = cosine_similarity(irled_query_data[query], para_vectors)
                sim_values = sim_values.reshape(-1)
                # msp = np.amax(sim_values, axis=1)

                sorted_index_array = np.argsort(sim_values)
                sorted_sim_values = sim_values[sorted_index_array]

                sim = np.mean(sorted_sim_values[-k_for_CAT : ])
                scores[doc] = sim 
                        
            else:
                scores[doc] = 0

        scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True)) 
        query_scores[query] = scores

    return query_scores

def store_ranklist(ranklist, filename):
    lines=[]
    for current_case in list(ranklist.keys()):
        docs = list(ranklist[current_case].keys())
        scores = list(ranklist[current_case].values())
        for doc in docs:
            lines.append(current_case[0:-4]+" Q0 "+doc[0:-4]+" "+str(docs.index(doc)+1)+' '+str(scores[docs.index(doc)])+" "+'para_level'+'\n')
    file = open('../data/results/'+filename,'w')
    file.writelines(lines)
    file.close()
def main():
    pickle_path = '../data/pickled_files'
    ik_pickle_path = '../../IK/data/pickled_files' 
    irled_query_data = load_pickle(pickle_path, 'irled_query_data')
    irled_prior_data = load_pickle(pickle_path, 'irled_prior_data')
    citation_anchor_text = load_pickle(pickle_path, 'citation_anchor_text')
    high_confidence_mapping_prior = load_pickle(ik_pickle_path, 'high_confidence_mapping_prior')

    values_of_k = list(range(1,11))
    for k_for_P in values_of_k:
        s1 = score_org_paragraphs(irled_prior_data, k_for_P, irled_query_data)
        f1 = 'P_'+str(k_for_P)
        store_ranklist(s1, f1)
        store_as_pickle(pickle_path, f1, s1)

    for k_for_CAT in values_of_k:
        s2 = score_citation_anchor_text(citation_anchor_text, k_for_CAT, irled_query_data, irled_prior_data, high_confidence_mapping_prior)
        f2 = 'CAT_'+str(k_for_CAT)
        store_ranklist(s2, f2)
        store_as_pickle(pickle_path, f2, s2)
if __name__ == '__main__':
    main() 