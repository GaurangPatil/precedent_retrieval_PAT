# Below imports are required for performing all operations in the class
"""
Reference: This code utilizes resources or concepts from the following repository
Repository: https://github.com/bhoomeendra/Paragraph_Resourcefulness
Author: Bhoomeendra Singh Sisodiya (GitHub username: bhoomeendra)
"""
import nltk
import os,re, pickle
from bs4 import BeautifulSoup
import copy,re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
STOPWORD = set(stopwords.words('english'))
pt = PorterStemmer()
class ProcessJudgments:
    def get_judgment_ids(directory):
        '''
        Takes the directory as input, returns the judgment IDs
        '''
        judgment_ids = []
    
        for filename in tqdm(os.listdir(directory),desc='Obtaining Judgment IDs'):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                judgment_ids.append(re.sub("\D", "", f))
        return judgment_ids
    def get_judgment_paths(directory):
        '''
        Takes the directory as input, returns the judgment paths
        '''
        paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        return paths
    def clean_original_paragraph(s):
        '''
        takes a paragraph as input, and replaces escape sequences and extra spaces by  single whitespace
        '''
        s = s.replace("\t",' ')
        s = s.replace('\n',' ')
        s = re.sub('\s\s+',' ',s)
        return s
    def replace_laws(text):
        """Making the section acts and articals into a single word"""
        regx_sections = "(?i)section.\s*\d+\w?(\s*\(\d+\))?\s*(\(\w\))?"
        regx_acts = r"(?i)\w+\sact(s)?[^\w]\s*(\d+)?"
        regx_articles = r"(?i)Article\s+(\w+)"
        text = re.sub(regx_sections,ProcessJudgments.preprocess_laws,text)
        text = re.sub(regx_acts,ProcessJudgments.preprocess_laws,text)
        text = re.sub(regx_articles,ProcessJudgments.preprocess_laws,text)
        return text
    def process_paragraph(s,stop_word=True):
        s = s.lower()
        s = s.replace("\t",' ')
        s = s.replace('\.','')
        s = s.replace('.','')
        s = s.replace('/-','')
        s = s.replace(',','')
        s = s.replace('(',' ')
        s = s.replace(')',' ')
        s = s.replace('\n',' ')
        s = s.replace(';','')
        s = s.replace('-','')
        s = s.replace("'",'')
        s = s.replace('"','')
        s = s.replace('@','')
        s = s.replace('%','')
        s = s.replace(':','')
        s = s.replace('"','')
        s = s.replace('/',' ')
        s = s.replace('"','')
        s = s.replace('"','')
        s = s.replace('[',' ')
        s = s.replace(']',' ')
        s = s.replace('=',' ')
        s = re.sub(r'\\',' ',s)
        s = re.sub('\s\s+',' ',s)
        
    
        
        word_count = 0
        if stop_word:
            s = re.sub(' +',' ',s)
            words = s.split(' ')
            output = ""
            for w in words:
                if w not in STOPWORD:
                    output+= pt.stem(w) +' '
                    word_count+=1
            if word_count >4:
                return output.strip()
            return ''
        
        return s
    def preprocess_laws(reMatchobj):
        text = reMatchobj.group(0)
        text.replace(' ','')
        text = ''.join(filter(str.isalnum, text))
        return " "+text+" "
    def html_to_text(html_file, html_folder):
        with open(html_folder+html_file, encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text()
        return text
    def convert_html_to_stemmed_text(html_folder):
        judgment_ids = ProcessJudgments.get_judgment_ids(html_folder)
        sc_judgments = {}
        for judgment in tqdm(judgment_ids, desc='Cleaning IK Judgments'):
            try:
                judgment_text = ProcessJudgments.html_to_text(html_file = '/'+judgment+'.html',html_folder = html_folder)
            except:
                with open(html_folder+'/'+judgment+'.txt', encoding='utf-8') as f:
                    doc = f.read()
                    doc = ProcessJudgments.replace_laws(doc)
                    doc = ProcessJudgments.process_paragraph(doc)
                    sc_judgments[judgment] = doc
                    continue
            judgment_text = ProcessJudgments.replace_laws(judgment_text)
            judgment_text = ProcessJudgments.process_paragraph(judgment_text)
            sc_judgments[judgment]=judgment_text
        return sc_judgments


def store_as_pickle(path, filename, object):
    pickle_path = path + '/'+ filename +'.pickle'
    with open(pickle_path, 'wb') as handle:
        pickle.dump(object, handle, protocol = pickle.HIGHEST_PROTOCOL)
    print(f'Dumped {filename} in {pickle_path}')

def load_pickle(path, filename):
    pickle_path = path + '/'+ filename +'.pickle'
    with open(pickle_path, 'rb') as handle:
        object = pickle.load(handle)
    return object