import os

from tqdm import tqdm
from utils import ProcessJudgments, store_as_pickle, load_pickle
import copy



def remove_current_cases(sc_citation_graph, mapping_current, mapping_current_2nd):
    current_cases_ik = [x[0] for x in mapping_current.values()]  + [x[0] for x in mapping_current_2nd.values() if x[1]>0.7]    
    sc_citation_graph_copy = copy.deepcopy(sc_citation_graph)
    # iterate over all cases that need to be removed 
    for ik_doc in current_cases_ik:
        # remove the case
        sc_citation_graph.pop(ik_doc, None)

        # iterate over citation graph
        for j, citations in sc_citation_graph_copy.items():
            # remove cases that cite the given case 
            if ik_doc in citations:
                sc_citation_graph.pop(j, None)
    return sc_citation_graph

def vectorize_irled_paragraphs_prior(vectorizer_filename, irled_docs_path):
    pickle_path = '../data/pickled_files'
    tfidf_vectorizer = load_pickle(pickle_path, vectorizer_filename)
    irled_judgment_data = {}
    irled_docs = os.listdir(irled_docs_path)
    for doc in tqdm(irled_docs, desc='Vectorizing Prior Case Paragraphs'):
        f = open(irled_docs_path+'/'+doc, "r",encoding='cp1252')
        paragraphs = f.readlines()
        cleaned_paragraphs = []
        for para in paragraphs:

            cleaned_para = ProcessJudgments.replace_laws(para)
            
            cleaned_para = ProcessJudgments.process_paragraph(cleaned_para)
            cleaned_paragraphs.append(cleaned_para)

        irled_judgment_data[doc] = tfidf_vectorizer.transform(cleaned_paragraphs if len(cleaned_paragraphs) else ['empty'])
    return irled_judgment_data
def vectorize_irled_queries(vectorizer_filename, irled_docs_path):
    pickle_path = '../data/pickled_files'
    tfidf_vectorizer = load_pickle(pickle_path, vectorizer_filename)
    irled_judgment_data = {}
    irled_docs = os.listdir(irled_docs_path)
    for doc in tqdm(irled_docs, desc='Vectorizing Query Case Citation Paragraphs'):
        f = open(irled_docs_path+'/'+doc, "r",encoding='cp1252')
        paragraphs = f.readlines()

        cleaned_paragraphs = []
        for para in paragraphs:

            if para.find('?citation?') > -1 or para.find('?CITATION?') > -1:
                cleaned_para = ProcessJudgments.replace_laws(para)
                
                cleaned_para = ProcessJudgments.process_paragraph(cleaned_para)
                cleaned_paragraphs.append(cleaned_para)

        if len(cleaned_paragraphs) == 0:
            print("No citations found:",doc)

            for para in paragraphs:
                cleaned_para = ProcessJudgments.replace_laws(para)
                    
                cleaned_para = ProcessJudgments.process_paragraph(cleaned_para)
                cleaned_paragraphs.append(cleaned_para)
        irled_judgment_data[doc] = tfidf_vectorizer.transform(cleaned_paragraphs if len(cleaned_paragraphs) else ['empty'])
    return irled_judgment_data

def main():

    pickle_path = '../data/pickled_files'
    ik_pickle_path = '../../IK/data/pickled_files'
   

    vectorizer_filename = 'tfidf_vectorizer_fitted_to_irled'
    irled_docs_path_prior = "../data/raw/Prior_Cases/"
    irled_docs_path_current = "../data/raw/Current_Cases/"


    mapping_current = load_pickle(ik_pickle_path, 'mapping_current_cases') 
    mapping_current_2nd = load_pickle(ik_pickle_path, 'mapping_current_cases_2nd')

    sc_citation_graph = load_pickle(ik_pickle_path, 'sc_citation_graph')

    sc_citation_graph = remove_current_cases(sc_citation_graph, mapping_current, mapping_current_2nd)
    store_as_pickle(ik_pickle_path, 'sc_citation_graph_exclude_queries', sc_citation_graph)

    print('Total judgments in new citation graph after removing current cases and cases that cite current cases:', len(sc_citation_graph))


    irled_prior_data = vectorize_irled_paragraphs_prior(vectorizer_filename, irled_docs_path_prior)

    store_as_pickle(pickle_path, 'irled_prior_data', irled_prior_data)
    irled_query_data = vectorize_irled_queries(vectorizer_filename, irled_docs_path_current)

    store_as_pickle(pickle_path, 'irled_query_data', irled_query_data)

if __name__ == '__main__':
    main() 

