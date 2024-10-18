from utils import ProcessJudgments, store_as_pickle, load_pickle
from tqdm import tqdm
import copy
import numpy as np
def get_mapping_aila_to_ik(aila_mapped_to_ik_vectors, remaining_ik_doc_vectors,aila_mapped_to_ik_names, remaining_ik_doc_names):
    mapping = {}
    for idx in tqdm(range(len(aila_mapped_to_ik_names)), desc='Mapping AILA query cases to similar judgments in ISCJD corpus'):
        similarity_values = aila_mapped_to_ik_vectors[idx].dot(remaining_ik_doc_vectors.T).toarray()
        index_corresponding_to_max_sim  = np.argmax(similarity_values)
        corresponding_similarity = similarity_values[0][index_corresponding_to_max_sim]
        corresponding_ik_doc = remaining_ik_doc_names[index_corresponding_to_max_sim]
        mapping[aila_mapped_to_ik_names[idx]] = (corresponding_ik_doc, corresponding_similarity)
    return mapping

sc_citation_graph = load_pickle('../../IK/data/pickled_files','sc_citation_graph')
ik_text = load_pickle('../../IK/data/pickled_files','ik_judgments_text')

aila_query_mapping_path = '../data/aila_query_mapping'
aila_query_mapping = open(aila_query_mapping_path, 'r').readlines()
aila_query_mapping = [val.strip('\n') for val in aila_query_mapping if val.strip('\n') !='-']
print('Total mapped by me (excluding dashes):',len(aila_query_mapping))
mapped = aila_query_mapping
remaining = list(set(sc_citation_graph.keys()).difference(set(mapped)))

vectorizer = load_pickle('../../IK/data/pickled_files','tfidf_vectorizer')
remaining_ik_doc_names= copy.copy(remaining)
remaining_ik_doc_vectors = vectorizer.transform([ik_text[val] for val in remaining])


mapped = [val for val in mapped if val in ik_text.keys()]
print('Total mapped by me and found in ISCJD:',len(mapped))
aila_mapped_to_ik_names = copy.copy(mapped)
aila_mapped_to_ik_vectors = vectorizer.transform([ik_text[val] for val in mapped])

similar_docs = get_mapping_aila_to_ik(aila_mapped_to_ik_vectors, remaining_ik_doc_vectors,aila_mapped_to_ik_names, remaining_ik_doc_names)
store_as_pickle('../data/pickled_files', 'aila_query_similar_docs', similar_docs)


aila_query_similar_docs = load_pickle('../data/pickled_files','aila_query_similar_docs')

greater = [(val,aila_query_similar_docs[val]) for val in aila_query_similar_docs if aila_query_similar_docs[val][1]>0.7]
to_be_removed = [aila_query_similar_docs[val][0] for val in aila_query_similar_docs if aila_query_similar_docs[val][1]>0.7]
for v in greater:
    print(v)
print('Docs similar to query docs : ',to_be_removed)