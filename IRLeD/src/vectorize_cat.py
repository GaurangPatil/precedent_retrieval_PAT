
from collections import defaultdict 
from tqdm import tqdm
from utils import store_as_pickle, load_pickle


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

def get_citation_anchor_text(required_ik_docs_are_cited_by,judgment_data, vectorizer_filename):
    pickle_path = '../data/pickled_files'
    tfidf_vectorizer = load_pickle(pickle_path, vectorizer_filename)
    citation_anchor_text = defaultdict(int)
    
    for irled_judgment, pointing_judgments in tqdm(required_ik_docs_are_cited_by.items(), desc='Retrieving and Vectorizing Citation Anchor Text (CAT)'):
        paras_list= []
        for judgment_id in pointing_judgments:
            for para in judgment_data[judgment_id].keys():
                if irled_judgment in judgment_data[judgment_id][para]['sc_judgments_cited']:
                    paras_list.append(judgment_data[judgment_id][para]['text'])
                    
        citation_anchor_text[irled_judgment] = tfidf_vectorizer.transform(paras_list if len(paras_list) else['empty'])
    return citation_anchor_text

def main():
    pickle_path = '../data/pickled_files'
    ik_pickle_path = '../../IK/data/pickled_files'

    vectorizer_filename = 'tfidf_vectorizer_fitted_to_irled'
    mapping_prior = load_pickle(ik_pickle_path,'mapping_prior_cases')

    judgment_data = load_pickle(ik_pickle_path, 'judgment_data')
    print('Total judgments in judgment_data:',len(judgment_data))
    high_confidence_mapping_prior = get_mapping_based_on_confidence(mapping_prior, threshold=0.9)
    store_as_pickle(ik_pickle_path, 'high_confidence_mapping_prior', high_confidence_mapping_prior)
    print('Number of judgments successfully mapped above the given probability threshold:',len(high_confidence_mapping_prior))
    sc_citation_graph_exclude_queries = load_pickle(ik_pickle_path, 'sc_citation_graph_exclude_queries')
    required_ik_docs_are_cited_by = get_pointing_judgments(high_confidence_mapping_prior, sc_citation_graph_exclude_queries)
    
    citation_anchor_text = get_citation_anchor_text(required_ik_docs_are_cited_by,judgment_data, vectorizer_filename)

    # storing paragraph level CAT vectors 
    store_as_pickle(pickle_path, 'citation_anchor_text', citation_anchor_text)
if __name__ == '__main__':
    main() 