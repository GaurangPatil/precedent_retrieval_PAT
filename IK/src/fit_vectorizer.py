from utils import ProcessJudgments

from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import store_as_pickle
def main():
    ISCJD_folder = '../data/raw/judgments'
    pickle_path = '../data/pickled_files'
    ik_judgments_text =  ProcessJudgments.convert_html_to_stemmed_text(ISCJD_folder)
    store_as_pickle(pickle_path, 'ik_judgments_text', ik_judgments_text)
    ik_docs_list = [doc for _,doc in ik_judgments_text.items()]

    print('Length of processed ISCJD judgments: ',len(ik_docs_list))

    tfidf_vectorizer = TfidfVectorizer(min_df=2,max_df=0.9,ngram_range=(1,2))
    tfidf_vectorizer.fit(ik_docs_list)
    store_as_pickle(pickle_path, 'tfidf_vectorizer', tfidf_vectorizer)

if __name__ == '__main__':
    main() 