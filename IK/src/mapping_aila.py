from utils import ProcessJudgments, load_pickle, store_as_pickle
import tqdm,os
from tqdm import tqdm
import numpy as np

def get_mapping_aila_to_ik(aila_doc_vectors, ik_doc_vectors,aila_doc_names, ik_doc_names):
    mapping = {}
    for idx in tqdm(range(len(aila_doc_names)), desc='Mapping AILA prior cases to their corresponding counterparts in ISCJD corpus'):
        similarity_values = aila_doc_vectors[idx].dot(ik_doc_vectors.T).toarray()
        index_corresponding_to_max_sim  = np.argmax(similarity_values)
        corresponding_similarity = similarity_values[0][index_corresponding_to_max_sim]
        corresponding_ik_doc = ik_doc_names[index_corresponding_to_max_sim]
        mapping[aila_doc_names[idx]] = (corresponding_ik_doc, corresponding_similarity)
    return mapping
def get_aila_docs_text(directory):
    aila_docs_text = {}
    for file in tqdm(os.listdir(directory),desc='Cleaning AILA documents'):         
        with open(directory + '/'+file, 'r') as f: 
            doc = f.read()
            doc = ProcessJudgments.replace_laws(doc)
            doc = ProcessJudgments.process_paragraph(doc)
            aila_docs_text[file] = doc
    return aila_docs_text
def main():
    ik_pickle_path = '../data/pickled_files'

    ik_judgments_text = load_pickle(ik_pickle_path, 'ik_judgments_text')

    ik_docs_list = [doc for _,doc in ik_judgments_text.items()]

    tfidf_vectorizer = load_pickle(ik_pickle_path,'tfidf_vectorizer')
    aila_prior_directory = '../../AILA/data/raw/prior_cases'

    aila_docs_text = get_aila_docs_text(aila_prior_directory)

    aila_doc_names = list(aila_docs_text.keys())
    aila_doc_vectors = tfidf_vectorizer.transform(list(aila_docs_text.values()))

    ik_doc_names = list(ik_judgments_text.keys())
    ik_doc_vectors = tfidf_vectorizer.transform(list(ik_docs_list))
    mapping_aila = get_mapping_aila_to_ik(aila_doc_vectors, 
                                    ik_doc_vectors,
                                    aila_doc_names, 
                                    ik_doc_names)
    store_as_pickle(ik_pickle_path,'mapping_aila',mapping_aila)

if __name__ == '__main__':
    main() 