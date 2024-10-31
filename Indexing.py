import collections
import numpy as np
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

N = 7
#nltk.download('stopwords')
#nltk.download('wordnet')


class Appearance:                              #Domh dedomenwn gia tis emfaniseis se ena document
    def __init__(self, docId, frequency):
        self.docId = docId
        self.frequency = frequency

    def __repr__(self):
        return str(self.__dict__)
    
class InvertedIndex:                           #Klash gia to eurethrio
    def __init__(self):
        self.index = dict()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def __repr__(self):
        return str(self.index)
    
    def index_document(self,document, doc_id): #Sinartisi me thn opoia dhmiourgeite to eurethrio
            word = document.read()             
            clean = self.clean_text(word)      #Katharizei tis lexeis diathrwntas tis simantikes
            terms = clean.split()              
            counter = collections.Counter(terms) #Counter poy ypologizei ta frequency gia kathe lexh entos listas
            
            for word in terms:                 #Gia kathe lexh sth lista an yparxei sto eyrethrio hdh elegxe to doc_id alliws prosthese thn 
                if word in self.index:         
                    ex = self.index[word]      #Ex exei th lista apo appearance ths kathe lexhs
                    inside = 1        
                    for item in ex:            #Gia kathe stoixeio ths listas an yparxei eisagwgh ths lexhs me ayto to docId sto eyrethrio kane inside = 0
                        if item.docId == doc_id:
                            inside = 0
                    if inside:                 #An den yparxei eisagwgh ths lexhs me ayto to doc_id prosthese thn
                        freq = counter[word]
                        app = Appearance(doc_id, freq)
                        self.index[word].append(app)
                else:                          
                    freq = counter[word]
                    app = Appearance(doc_id, freq)
                    apps = []
                    apps.append(app)
                    self.index[word] = apps
    
    def clean_text(self, text):               #Sinartisi poy metatrepei tis lexeis se eniko me mikra grammata kai afairei ta stopwords apo to keimeno 
        stop_words = set(stopwords.words('english'))
        clean_text = re.sub(r'[^\w\s]', '', text)
        terms = clean_text.split()
        filtered = [self.lemmatizer.lemmatize(self.stemmer.stem(term.lower())) for term in terms if term.lower() not in stop_words]
        return ' '.join(filtered)
        
index = InvertedIndex()

for doc_id in range(1, N):                    #Gia ola ta arxeia me id sto range twn dyo arithmwn (An range(1,3) tote arxeia me id 1 kai 2)
    file_path = 'docs/'+f'{doc_id:05}'+'.txt' #Metetrepse to id se filepath (An id = 1 tote filepath 00001.txt)
    f = open(file_path, 'r')                  #Anoixe to arxeio
    
    index.index_document(f, doc_id)
    
    f.close()                                 #Kleise to arxeio
    
print(index)