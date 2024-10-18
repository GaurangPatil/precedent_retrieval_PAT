from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from utils import store_as_pickle, load_pickle
def store_ranklist(ranklist, filename):
    lines=[]
    for current_case in list(ranklist.keys()):
        docs = list(ranklist[current_case].keys())
        scores = list(ranklist[current_case].values())
        for doc in docs:
            lines.append(current_case[0:-4]+" Q0 "+doc[0:-4]+" "+str(docs.index(doc)+1)+' '+str(scores[docs.index(doc)])+" "+'doc_level_cat'+'\n')

    file = open('../data/results/'+filename,'w')
    file.writelines(lines)
    file.close()
def vectorize_judments(docs, vectorizer_filename):
    pickle_path = '../data/pickled_files'
    tfidf_vectorizer = load_pickle(pickle_path, vectorizer_filename)
    vectors = {}
    for doc in tqdm(docs, desc='Vectorizing Case Docs'):
        vectors[doc] = tfidf_vectorizer.transform([docs[doc]] if len(docs[doc]) else ['empty'])
    return vectors

def score_doc_level_cat(cat_vectors, current_vectors, prior_vectors_keys):
    query_scores = {}
    # iterate over all queries
    for query in tqdm(current_vectors.keys(),desc=f'Computing similarity at Doc level using CAT alone'):

        scores = {}
        # iterate over prior cases 
        for doc in prior_vectors_keys:
            try: # if CAT is available for the doc
                sim_value = cosine_similarity(current_vectors[query], cat_vectors[doc])
                
                scores[doc] = sim_value[0][0]
            except:
                scores[doc] = 0
        scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True)) 
        query_scores[query] = scores
    return query_scores
def main():
    pickle_path = '../data/pickled_files'
    vectorizer_filename = 'tfidf_vectorizer_fitted_to_irled'
    citation_anchor_text_doc_level = load_pickle(pickle_path, 'citation_anchor_text_doc_level')
    irled_docs_text_current = load_pickle(pickle_path, 'irled_docs_text_current')
    prior_vectors_keys = load_pickle(pickle_path, 'prior_vectors').keys()

    cat_vectors = vectorize_judments(citation_anchor_text_doc_level, vectorizer_filename)
    current_vectors = vectorize_judments(irled_docs_text_current, vectorizer_filename)
    store_as_pickle(pickle_path,'cat_vectors', cat_vectors)
    store_as_pickle(pickle_path,'current_vectors', current_vectors)
    s1 = score_doc_level_cat(cat_vectors, current_vectors, prior_vectors_keys)
    f1 = 'doc_level_cat'
    store_ranklist(s1, f1)
if __name__ == '__main__':
    
    main() 