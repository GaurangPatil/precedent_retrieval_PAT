
from tqdm import tqdm
import copy
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from utils import store_as_pickle, load_pickle
import scipy.sparse as sp
def store_ranklist(ranklist, filename):
    lines=[]
    for current_case in list(ranklist.keys()):
        docs = list(ranklist[current_case].keys())
        scores = list(ranklist[current_case].values())
        for doc in docs:
            lines.append(current_case[0:-4]+" Q0 "+doc[0:-4]+" "+str(docs.index(doc)+1)+' '+str(scores[docs.index(doc)])+" "+'para_level_p_union_cat'+'\n')
    file = open('../data/results/'+filename,'w')
    file.writelines(lines)
    file.close()
def score_p_union_cat(irled_judgment_data_prior, k_for_P_union_CAT, irled_query_data, high_confidence_mapping_prior, citation_anchor_text):
    query_scores = {}
    # iterate over all queries
    for query in tqdm(irled_query_data.keys(),desc=f'Computing similarity using P U CAT with k = {k_for_P_union_CAT}'):

        scores = {}
        for doc in irled_judgment_data_prior.keys():

            # get original paragraphs of prior case
            P = irled_judgment_data_prior[doc]
            final_representation = copy.deepcopy(P)

            # if case is mapped with high confidence 
            if doc in high_confidence_mapping_prior.keys():
                # get CAT for prior case 
                CAT = citation_anchor_text[high_confidence_mapping_prior[doc][0]]
                # take union of all vector representations of P and CAT
                final_representation = sp.vstack((P,CAT))
            sim_values = cosine_similarity(irled_query_data[query], final_representation)
            
            sim_values = sim_values.reshape(-1)
            
            sorted_index_array = np.argsort(sim_values)
            sorted_sim_values = sim_values[sorted_index_array]

            sim = np.mean(sorted_sim_values[-k_for_P_union_CAT : ])
            scores[doc] = sim
            
        scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True)) 
        query_scores[query] = scores

        
    return query_scores

def main():
    pickle_path = '../data/pickled_files'
    ik_pickle_path = '../../IK/data/pickled_files' 
    irled_query_data = load_pickle(pickle_path, 'irled_query_data')
    irled_prior_data = load_pickle(pickle_path, 'irled_prior_data')
    citation_anchor_text = load_pickle(pickle_path, 'citation_anchor_text')
    high_confidence_mapping_prior = load_pickle(ik_pickle_path, 'high_confidence_mapping_prior')

    values_of_k = list(range(1,11))
    for k_for_P_union_CAT in values_of_k:
        s1 = score_p_union_cat(irled_prior_data, k_for_P_union_CAT, irled_query_data, high_confidence_mapping_prior, citation_anchor_text)
        f1 = 'P_U_CAT_'+str(k_for_P_union_CAT)
        store_ranklist(s1, f1)
        store_as_pickle(pickle_path, f1, s1)

if __name__ == '__main__':
    main() 