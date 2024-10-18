from utils import ProcessJudgments, load_pickle, store_as_pickle
import tqdm,os,copy
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm

def find_second_highest_index(sim_values):
    
    highest = -np.inf
    second_highest = -np.inf
    
    # Find the highest and second highest values
    for value in sim_values:
        if value > highest:
            second_highest = copy.copy(highest)
            highest = copy.copy(value)
        elif value > second_highest and value < highest:
            second_highest = copy.copy(value)
    
    # get index of the second highest value
    second_highest_index = np.where(sim_values == second_highest)[0][0]
    
    return second_highest_index

def get_irled_docs_text(directory):
    irled_docs_text = {}
    for file in tqdm(os.listdir(directory), desc='Cleaning IRLeD Current Cases Text'):         
        with open(directory+'/'+file, 'r',encoding='cp1252') as f: 
            doc = f.read()
            doc = ProcessJudgments.replace_laws(doc)
            doc = ProcessJudgments.process_paragraph(doc)
            irled_docs_text[file] = doc
    return irled_docs_text

def get_mapping_irled_to_ik(irled_doc_vectors, ik_doc_vectors,irled_doc_names, ik_doc_names):
    mapping = {}
    mapping_2nd = {}
    for idx in tqdm(range(len(irled_doc_names)),desc='Mapping Current Cases to their counterparts in ISCJD corpus'):
        similarity_values = irled_doc_vectors[idx].dot(ik_doc_vectors.T).toarray()
        index_corresponding_to_max_sim  = np.argmax(similarity_values)
        corresponding_similarity = similarity_values[0][index_corresponding_to_max_sim]
        corresponding_ik_doc = ik_doc_names[index_corresponding_to_max_sim]
        mapping[irled_doc_names[idx]] = (corresponding_ik_doc, corresponding_similarity)
        index_corresponding_to_max_sim_2nd = find_second_highest_index(similarity_values[0])
        corresponding_similarity_2nd = similarity_values[0][index_corresponding_to_max_sim_2nd]
        corresponding_ik_doc_2nd = ik_doc_names[index_corresponding_to_max_sim_2nd]
        mapping_2nd[irled_doc_names[idx]] = (corresponding_ik_doc_2nd, corresponding_similarity_2nd)
    return mapping, mapping_2nd

def main():
    pickle_path = '../data/pickled_files'
    ik_judgments_text = load_pickle(pickle_path, 'ik_judgments_text')
    tfidf_vectorizer = load_pickle(pickle_path,'tfidf_vectorizer')
    ik_docs_list = [doc for _,doc in ik_judgments_text.items()]
    ik_doc_names = list(ik_judgments_text.keys())
    ik_doc_vectors = tfidf_vectorizer.transform(ik_docs_list)

    irled_current_directory = '../../IRLeD/data/raw/Current_Cases'

    irled_docs_text = get_irled_docs_text(irled_current_directory)
    irled_doc_names = list(irled_docs_text.keys())

    irled_doc_vectors = tfidf_vectorizer.transform(list(irled_docs_text.values()))
    mapping, mapping_2nd = get_mapping_irled_to_ik(irled_doc_vectors, 
                                    ik_doc_vectors,
                                    irled_doc_names, 
                                    ik_doc_names)
    store_as_pickle(pickle_path,'mapping_current_cases',mapping)
    store_as_pickle(pickle_path,'mapping_current_cases_2nd',mapping_2nd)
if __name__ == '__main__':

    main()