
from tqdm import tqdm
from utils import ProcessJudgments,load_pickle,store_as_pickle
from statistics import mean 

from sklearn.metrics.pairwise import cosine_similarity

def store_ranklist(ranklist, filename):
    lines=[]
    for current_case in list(ranklist.keys())[10:]:
        docs = list(ranklist[current_case].keys())
        scores = list(ranklist[current_case].values())
        for doc in docs:
            lines.append(current_case+" Q0 "+doc[0:-4]+" "+str(docs.index(doc)+1)+' '+str(scores[docs.index(doc)])+" "+'doc_level_cat'+'\n')
    file = open('../data/results/'+filename,'w')
    file.writelines(lines)
    file.close()
def transform_queries(query_file_path, vectorizer_filename):
    pickle_path = '../data/pickled_files'
    tfidf_vectorizer = load_pickle(pickle_path, vectorizer_filename)

    query_vectors = {}

    with open(query_file_path, 'r') as f:
        queries = f.readlines()
    i = 1
    for query in tqdm(queries, desc='Vectorizing AILA Queries'):
        cleaned_query = ProcessJudgments.replace_laws(query)
        cleaned_query = ProcessJudgments.process_paragraph(cleaned_query)
        query_vectors['AILA_Q'+ str(i)] = (query,tfidf_vectorizer.transform([cleaned_query]))
        i += 1

    return query_vectors
def score_doc_level_cat(q_vectors, citation_anchor_text_doc_level_vectors, file_names):
    query_scores = {}
    for q_id, q_vector in tqdm(q_vectors.items(), desc=f'Computing similarity at Doc level using CAT alone'):
        # iterate over all prior docs
        scores = {}
        for doc in file_names:
            try: # If CAT available 
                sim = cosine_similarity(q_vector[1], citation_anchor_text_doc_level_vectors[doc])[0][0]
                scores[doc] = sim
            except:
                scores[doc] = 0
        scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True)) 
        query_scores[q_id] = scores
    return query_scores

def main():
    vectorizer_name = 'tfidf_vectorizer_fitted_to_aila'
    query_file_path = '../data/raw/query_doc.txt'
    pickle_path = '../data/pickled_files'
    citation_anchor_text_doc_level_vectors = load_pickle(pickle_path, 'citation_anchor_text_doc_level_vectors')

    query_vectors = transform_queries(query_file_path,vectorizer_name)
    
    aila_judgment_doc_vectors = load_pickle(pickle_path, 'aila_judgment_doc_vectors')
    s1 = score_doc_level_cat(query_vectors,  citation_anchor_text_doc_level_vectors, aila_judgment_doc_vectors.keys())
    f1 = 'doc_level_cat'


    store_ranklist(s1, f1)
    store_as_pickle(pickle_path, f1, s1)
if __name__=='__main__':

    main()
