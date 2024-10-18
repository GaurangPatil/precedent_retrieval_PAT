import nltk
import os,re,pickle
from bs4 import BeautifulSoup
import copy,re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
from utils import ProcessJudgments, store_as_pickle
nltk.download('stopwords')
STOPWORD = set(stopwords.words('english'))
pt = PorterStemmer()
judgment_ids = [] 
def get_judgment_data(soup):
    global judgment_ids
    sc_cited = []
    other_cited = []
    para_num = 1
    citation_graph = {}
    paragraphs = list(soup.find_all("p"))

    # iterate over paragraphs in the judgment
    for para in paragraphs:
        sc_judgments_cited = []
        sc_judgments_cited_names = []
        other_judgments_cited = []
        other_judgments_cited_names = []

        citation_graph[para_num]={}
        paragraph = copy.deepcopy(para.text)
        original_para = ProcessJudgments.clean_original_paragraph(paragraph)
        paragraph = paragraph.lower()
        
        # find hyperlinks in the paragraph
        for a in para.find_all('a', href=True):
            identifier = re.sub("\D", "", a['href'])

            if a.string==None:
                continue
            replacestr = copy.deepcopy(a.string.replace("\t",""))
            replacestr = replacestr.replace("\n",'')
            replacestr = replacestr.replace(" ",'')

            # if link points to a judgment
            if 'v.' in replacestr or 'vs.' or ' v ' or ' vs ' in replacestr: 
                # if it's a SC judgment
                if identifier in judgment_ids: 
                    sc_judgments_cited.append(identifier)
                    sc_judgments_cited_names.append(replacestr)
                    sc_cited.append(identifier)
                else:
                    other_judgments_cited.append(identifier)
                    other_judgments_cited_names.append(replacestr)
                    other_cited.append(identifier)
                


        citation_graph[para_num]['sc_judgments_cited'] = sc_judgments_cited
        citation_graph[para_num]['other_judgments_cited'] = other_judgments_cited
        citation_graph[para_num]['sc_judgments_cited_names'] = sc_judgments_cited_names
        citation_graph[para_num]['other_judgments_cited_names'] = other_judgments_cited_names
        paragraph = ProcessJudgments.replace_laws(paragraph)
        paragraph = ProcessJudgments.process_paragraph(paragraph)

        citation_graph[para_num]['text'] = paragraph
        citation_graph[para_num]['org_text'] = original_para
        para_num += 1
    return sc_cited, other_cited, citation_graph

def get_ik_data(judgment_paths):
    judgment_data = {}
    sc_citations = {}
    other_citations = {}

    for judgment_path in tqdm(judgment_paths, desc='Extracting Data From Judgments Present in ISCJD'):

        with open(judgment_path, encoding='utf-8') as fp:
            soup = BeautifulSoup(fp, "html.parser")
            j_id = re.sub("\D", "", judgment_path)
            sc_citations[j_id],other_citations[j_id],judgment_data[j_id] = get_judgment_data(soup)
    return sc_citations,other_citations, judgment_data 

def main():
    global judgment_ids
    directory = '../data/raw/judgments'
    judgment_ids = ProcessJudgments.get_judgment_ids(directory)
    judgment_paths = ProcessJudgments.get_judgment_paths(directory)
    sc_citations, other_citations, judgment_data = get_ik_data(judgment_paths)
    print('Total Judgments processed : ', len(judgment_data.keys()))

    path = '../data/pickled_files'
    store_as_pickle(path,'judgment_data', judgment_data)
    store_as_pickle(path,'sc_citation_graph', sc_citations)
    store_as_pickle(path,'other_citation_graph', other_citations)

if __name__ == '__main__':
    main() 


