
from tqdm import tqdm
from utils import ProcessJudgments,load_pickle,store_as_pickle
import os


def vectorize_aila_paragraphs(vectorizer, aila_docs_path):

    pickle_path = '../data/pickled_files'
    tfidf_vectorizer = load_pickle(pickle_path,vectorizer)
    aila_judgment_data = {}
    aila_docs = os.listdir(aila_docs_path)
    for doc in tqdm(aila_docs):
        f = open(aila_docs_path+'/'+doc, "r")

        paragraphs = f.readlines()
       
        cleaned_paras = []
        for para in paragraphs:
            cleaned_para = ProcessJudgments.replace_laws(para)
            cleaned_para = ProcessJudgments.process_paragraph(cleaned_para)
            cleaned_paras.append(cleaned_para)
        aila_judgment_data[doc] = tfidf_vectorizer.transform(cleaned_paras)
    return aila_judgment_data
def main():
    pickle_path = '../data/pickled_files'
    vectorizer = 'tfidf_vectorizer_fitted_to_aila'
    aila_docs_path = '../data/raw/prior_cases'
    aila_prior_data = vectorize_aila_paragraphs(vectorizer, aila_docs_path)
    store_as_pickle(pickle_path,'aila_prior_data', aila_prior_data)
if __name__ == '__main__':
    main()