from utils import store_as_pickle, load_pickle
from collections import Counter

def store_ranklist(ranklist, run_id):
    lines=[]
    for current_case in list(ranklist.keys()):
        docs = list(ranklist[current_case].keys())
        scores = list(ranklist[current_case].values())
        for doc in docs:
            lines.append(current_case[0:-4]+" Q0 "+doc[0:-4]+" "+str(docs.index(doc)+1)+' '+str(scores[docs.index(doc)])+" "+'para_level_combined'+'\n')
    filename = run_id 
    file = open('../data/results/'+filename,'w')
    file.writelines(lines)
    file.close()
def combine_ranklists(ranklist_P, ranklist_CAT, alpha, beta):
    newdict= {}
    for current_case in list(ranklist_P.keys()):
        newdict[current_case] = dict(Counter({key: value *beta for key, value in ranklist_CAT[current_case].items()}) + Counter({key: value *alpha for key, value in ranklist_P[current_case].items()}))
        newdict[current_case] = dict(sorted(newdict[current_case].items(), key=lambda item: item[1],reverse=True))
    return newdict
def main():
    pickle_path = '../data/pickled_files'
    alpha_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    s1 = load_pickle(pickle_path, 'P_4')
    s2 = load_pickle(pickle_path, 'CAT_2')
    for alpha in alpha_values:
        beta = 1 - alpha
        final_ranklist = combine_ranklists(s1,s2, alpha, beta)
        filename = 'P_and_CAT_alpha_' + str(int(alpha * 10))
        store_as_pickle(pickle_path, filename, final_ranklist)
        store_ranklist(final_ranklist,filename )
if __name__ == '__main__':
    main() 