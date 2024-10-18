from collections import defaultdict 
from tqdm import tqdm
from utils import load_pickle,store_as_pickle
import copy


def get_mapping_based_on_confidence(mapping, threshold):
    
    high_confidence_mapping = defaultdict(int)
    for aila_doc, corresponding_ik_doc_and_similarity in mapping.items():
        if corresponding_ik_doc_and_similarity[1]>threshold: 
            high_confidence_mapping[aila_doc] = corresponding_ik_doc_and_similarity
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
    
    for aila_judgment, pointing_judgments in tqdm(required_ik_docs_are_cited_by.items(),desc='Retrieving and Vectorizing Citation Anchor Text (CAT)'):
        paras_list= []
        for judgment_id in pointing_judgments:
            for para in judgment_data[judgment_id].keys():
                if aila_judgment in judgment_data[judgment_id][para]['sc_judgments_cited']:
                    paras_list.append(judgment_data[judgment_id][para]['text'])
        if len(paras_list):
            citation_anchor_text[aila_judgment] = tfidf_vectorizer.transform(paras_list)
    return citation_anchor_text

def remove_queries(sc_citation_graph, aila_query_mapping):

    sc_citation_graph_copy = copy.deepcopy(sc_citation_graph)
    for ik_doc in aila_query_mapping:

        if ik_doc not in sc_citation_graph.keys():
            print('Not present:', ik_doc)
        # remove query cases
        sc_citation_graph.pop(ik_doc, None)
        for j, citations in sc_citation_graph_copy.items():
            # remove cases that cite the given case 
            if ik_doc in citations:
                sc_citation_graph.pop(j, None)
    return sc_citation_graph
def main():
    pickle_path = '../data/pickled_files'
    ik_pickle_path = '../../IK/data/pickled_files'
    vectorizer_name = 'tfidf_vectorizer_fitted_to_aila'
    mapping_aila = load_pickle(ik_pickle_path, 'mapping_aila')
    high_confidence_mapping_aila = get_mapping_based_on_confidence(mapping_aila, threshold=0.9)
    store_as_pickle(ik_pickle_path, 'high_confidence_mapping_aila', high_confidence_mapping_aila)
    sc_citation_graph = load_pickle(ik_pickle_path, 'sc_citation_graph')
    judgment_data = load_pickle(ik_pickle_path, 'judgment_data')
    aila_query_mapping_path = '../data/aila_query_mapping'
    aila_query_mapping = open(aila_query_mapping_path, 'r').readlines()
    aila_query_mapping = [val.strip('\n') for val in aila_query_mapping]
    aila_query_similar_path = '../data/aila_query_similar'
    aila_query_similar = open(aila_query_similar_path, 'r').readlines()
    aila_query_similar = [val.strip('\n') for val in aila_query_similar]
 
    print('Other judgments similar to AILA queries:',len(aila_query_similar))
    sc_citation_graph_exclude_aila_queries =  remove_queries(sc_citation_graph, aila_query_mapping + aila_query_similar) 
    print('Total documents in ISCJD after removing queries and cases citing the queries: ',len(sc_citation_graph_exclude_aila_queries))
    required_ik_docs_are_cited_by = get_pointing_judgments(high_confidence_mapping_aila, sc_citation_graph_exclude_aila_queries)
    citation_anchor_text = get_citation_anchor_text(required_ik_docs_are_cited_by,judgment_data, vectorizer_name )
    print('len of CAT:',len(citation_anchor_text))
    store_as_pickle(pickle_path, 'citation_anchor_text', citation_anchor_text)
    store_as_pickle(pickle_path, 'sc_citation_graph_exclude_aila_queries', sc_citation_graph_exclude_aila_queries)
if __name__=='__main__':
    main()