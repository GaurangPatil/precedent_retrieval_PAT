from tqdm import tqdm


from sklearn.metrics.pairwise import cosine_similarity
from utils import store_as_pickle, load_pickle
def store_ranklist(ranklist, filename):
    lines=[]
    for current_case in list(ranklist.keys()):
        docs = list(ranklist[current_case].keys())
        scores = list(ranklist[current_case].values())
        for doc in docs:
            lines.append(current_case[0:-4]+" Q0 "+doc[0:-4]+" "+str(docs.index(doc)+1)+' '+str(scores[docs.index(doc)])+" "+'doc_level_cat_appended'+'\n')
    file = open('../data/results/'+filename,'w')
    file.writelines(lines)
    file.close()
def vectorize_judgments(docs, vectorizer_filename):
    pickle_path = '../data/pickled_files'
    tfidf_vectorizer = load_pickle(pickle_path, vectorizer_filename)
    vectors = {}
    for doc in tqdm(docs, desc='Vectorizing Docs'):
        vectors[doc] = tfidf_vectorizer.transform([docs[doc]] if len(docs[doc]) else ['empty'])
    return vectors

def score_doc_level(appended_text_vectors, current_vectors):
    query_scores = {}
    # iterate over all queries
    for query in tqdm(current_vectors.keys(),desc=f'Computing similarity at Doc level by appending CAT'):

        scores = {}
        # iterate over all prior docs
        for doc in appended_text_vectors.keys():

            sim_value = cosine_similarity(current_vectors[query], appended_text_vectors[doc])
            
            scores[doc] = sim_value[0][0]
            
        scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True)) 
        query_scores[query] = scores
    return query_scores
def append_CAT_to_prior_case(prior_text, cat):
    appended_text = {}
    for doc in prior_text:
        try:
            appended_text[doc] = prior_text[doc] + cat[doc]
        except:
            appended_text[doc] = prior_text[doc]
    return appended_text

def main():
    pickle_path = '../data/pickled_files'
    vectorizer_filename = 'tfidf_vectorizer_fitted_to_irled'
    citation_anchor_text_doc_level = load_pickle(pickle_path, 'citation_anchor_text_doc_level')

    irled_docs_text_prior = load_pickle(pickle_path, 'irled_docs_text_prior')
    irled_docs_text_current = load_pickle(pickle_path, 'irled_docs_text_current')
    appended_text = append_CAT_to_prior_case(irled_docs_text_prior, citation_anchor_text_doc_level)

    appended_text_vectors = vectorize_judgments(appended_text, vectorizer_filename)
    current_vectors = vectorize_judgments(irled_docs_text_current, vectorizer_filename)
    store_as_pickle(pickle_path,'appended_text_vectors', appended_text_vectors)
    store_as_pickle(pickle_path,'current_vectors', current_vectors)
    s1 = score_doc_level(appended_text_vectors, current_vectors)
    f1 = 'doc_level_cat_appended'
    store_ranklist(s1, f1)
if __name__ == '__main__':    
    main() 