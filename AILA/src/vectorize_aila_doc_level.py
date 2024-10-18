
from tqdm import tqdm
from utils import ProcessJudgments,load_pickle,store_as_pickle
from statistics import mean 
import os


def vectorize_aila_doc_level(vectorizer, aila_docs_path):

    pickle_path = '../data/pickled_files'
    tfidf_vectorizer = load_pickle(pickle_path,vectorizer)
    aila_judgment_doc_vectors = {}
    aila_judgment_text_doc_level = {}
    aila_docs = os.listdir(aila_docs_path)
    for doc in tqdm(aila_docs, desc='Vectorizing AILA documents at the doc level'):
        
        f = open(aila_docs_path+'/'+doc, "r")

        doc_text = f.read()
       
        
        doc_text = ProcessJudgments.replace_laws(doc_text)
            
        doc_text = ProcessJudgments.process_paragraph(doc_text)

        aila_judgment_doc_vectors[doc] = tfidf_vectorizer.transform([doc_text if len(doc_text) else ['empty']])
        aila_judgment_text_doc_level[doc] = doc_text
    return aila_judgment_text_doc_level, aila_judgment_doc_vectors
def main():
    pickle_path = '../data/pickled_files'
    vectorizer = 'tfidf_vectorizer_fitted_to_aila'
    aila_docs_path = '../data/raw/prior_cases'
    aila_judgment_text_doc_level, aila_judgment_doc_vectors = vectorize_aila_doc_level(vectorizer, aila_docs_path)
    store_as_pickle(pickle_path,'aila_judgment_text_doc_level', aila_judgment_text_doc_level)
    store_as_pickle(pickle_path, 'aila_judgment_doc_vectors', aila_judgment_doc_vectors)
if __name__ == '__main__':
    main()