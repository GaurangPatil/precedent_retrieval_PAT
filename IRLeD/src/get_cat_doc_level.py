
from collections import defaultdict 
from tqdm import tqdm
from utils import  store_as_pickle, load_pickle
import copy


def get_mapping_based_on_confidence(mapping, threshold):
    
    high_confidence_mapping = defaultdict(int)
    for irled_doc, corresponding_ik_doc_and_similarity in mapping.items():
        if corresponding_ik_doc_and_similarity[1]>threshold: 
            high_confidence_mapping[irled_doc] = corresponding_ik_doc_and_similarity
    return high_confidence_mapping
def get_pointing_judgments(mapping, citation_graph):
    cited_by = defaultdict(int)
    required_docs = mapping.values() 
    required_docs = list(zip(*required_docs))[0]
    for doc in tqdm(required_docs):
        cited_by_list = []
        for sc_judgment, cited_judgments_in_sc_judgment in citation_graph.items():
            if doc in cited_judgments_in_sc_judgment:
                cited_by_list.append(sc_judgment)
        cited_by[doc] = cited_by_list 
    return cited_by
def get_citation_anchor_text_doc_level(required_ik_docs_are_cited_by,judgment_data, reverse_mapping):
    citation_anchor_text_doc_level = defaultdict(int)
    
    for irled_judgment, pointing_judgments in tqdm(required_ik_docs_are_cited_by.items(), desc='Retrieving Citation Anchor Text (CAT)'):
        paras_combined = ''
        for judgment_id in pointing_judgments:
            for para in judgment_data[judgment_id].keys():
                if irled_judgment in judgment_data[judgment_id][para]['sc_judgments_cited']:
                    paras_combined += ' '+(judgment_data[judgment_id][para]['text'])
                    
        citation_anchor_text_doc_level[reverse_mapping[irled_judgment]] = paras_combined if len(paras_combined) else ''
    return citation_anchor_text_doc_level

def main():
    pickle_path = '../data/pickled_files'
    ik_pickle_path = '../../IK/data/pickled_files'

    mapping_prior = load_pickle(ik_pickle_path,'mapping_prior_cases')

    judgment_data = load_pickle(ik_pickle_path, 'judgment_data')
    print('Total judgments in judgment_data:',len(judgment_data))
    high_confidence_mapping_prior = get_mapping_based_on_confidence(mapping_prior, threshold=0.9)
    store_as_pickle(ik_pickle_path, 'high_confidence_mapping_prior', high_confidence_mapping_prior)
    print('Number of judgments successfully mapped abbove the given probability threshold:',len(high_confidence_mapping_prior))
    sc_citation_graph_exclude_queries = load_pickle(ik_pickle_path, 'sc_citation_graph_exclude_queries')
    required_ik_docs_are_cited_by = get_pointing_judgments(high_confidence_mapping_prior, sc_citation_graph_exclude_queries)
    reverse_mapping = {}

    for key, (value_str, _) in high_confidence_mapping_prior.items():
        new_key = copy.deepcopy(value_str)
        new_value = copy.deepcopy(key)
        reverse_mapping[new_key] = new_value
    citation_anchor_text_doc_level = get_citation_anchor_text_doc_level(required_ik_docs_are_cited_by,
                                                                        judgment_data, 
                                                                        reverse_mapping)
    
    store_as_pickle(pickle_path, 'citation_anchor_text_doc_level', citation_anchor_text_doc_level)
if __name__ == '__main__':
    main() 