import collections
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

N = 1240
# nltk.download('stopwords')
# nltk.download('wordnet')

#Domh dedomenwn gia tis emfaniseis se ena document
class Appearance: 
    def __init__(self, docId, frequency):
        self.docId = docId
        self.frequency = frequency

    def __repr__(self):
        return f"[docId: {self.docId}, frequency: {self.frequency}]"

#Klash gia to eurethrio
class InvertedIndex:
    def __init__(self):
        self.index = dict()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def __repr__(self):
        result = ""
        for word, appearances in self.index.items():
            result += f"{word}: "
            result += f"{appearances[0]}"
            for appearance in appearances[1:]:
                result += f", {appearance}"
            result += "\n"
        return result
    
    #Sinartisi me thn opoia dhmiourgeite to eurethrio
    def index_document(self, document, doc_id):
        clean = self.clean_text(document) #Katharizei tis lexeis diathrwntas tis simantikes
        terms = clean.split()
        counter = collections.Counter(terms) #Counter poy ypologizei ta frequency gia kathe lexh entos listas

        for word in terms: #Gia kathe lexh sth lista an yparxei sto eyrethrio hdh elegxe to doc_id alliws prosthese thn 
            if word in self.index:
                ex = self.index[word] #Ex exei th lista apo appearance ths kathe lexhs
                inside = 1
                for item in ex: #Gia kathe stoixeio ths listas an yparxei eisagwgh ths lexhs me ayto to docId sto eyrethrio kane inside = 0
                    if item.docId == doc_id:
                        inside = 0  
                if inside: #An den yparxei eisagwgh ths lexhs me ayto to doc_id prosthese thn
                    freq = counter[word]
                    app = Appearance(doc_id, freq)
                    self.index[word].append(app)
            else:
                freq = counter[word]
                app = Appearance(doc_id, freq)
                apps = []
                apps.append(app)
                self.index[word] = apps

    #Sinartisi poy metatrepei tis lexeis se eniko me mikra grammata kai afairei ta stopwords apo to keimeno 
    def clean_text(self, text):
        stop_words = set(stopwords.words('english'))
        clean_text = re.sub(r'[^\w\s]', '', text)
        terms = clean_text.split()
        filtered = [self.lemmatizer.lemmatize(self.stemmer.stem(term.lower())) for term in terms if term.lower() not in stop_words]
        return ' '.join(filtered)

    #Sinartisi poy dhmiourgei to vector space montelo
    def vector_space(self, N):
        rows = len(self.index) + 1
        columns = N
        w_matrix = [[0 for i in range(columns)] for j in range(rows)] #Pinakas ston opoion tha vriskontai ta varh kathe lexhs ana keimeno
        w_matrix[0][0] = "Word\Doc id"

        for i in range(1, N): #Sthn grammh 0 toy pinaka yparxei index gia ta doc id
            w_matrix[0][i] = i

        i = 1
        for word in self.index: #Sthn sthlh 0 toy pinaka anagrafontai oi lexeis
            w_matrix[i][0] = word
            i += 1

        for i in range(1, rows): #Se kathe allo stoixeio toy pinaka ypologizontai ta katallhla tf-idf gia thn kathe lexh sto ekastote keimeno (Gia thn lexh sth thesh [1][0] h thesh [1][1] apothikeyei to varos ths sto doc 1)
            idf_list = self.index[w_matrix[i][0]]
            ni = len(idf_list)
            idf = math.log2(1 + ((N - 1) / ni))
            for item in idf_list:
                f_ij = item.frequency
                j = item.docId
                tf = 1 + math.log2(f_ij)
                result = round(tf * idf, 3)
                w_matrix[i][j] = result
        return w_matrix

    #Sinartisi poy epistrefei ta varh twn keimenwn synolika ws dianismata
    def get_vectors(self, matrix, N): 
        d = []
        d.append("Vectors:")
        for j in range(1, N): #Dhmiourgeitai lista sthn opoia apothikeyontai ta varh enos keimenoy kai meta afth h lista metatrpetai se dianisma
            lst = []
            for i in range(1, len(matrix)):
                lst.append(matrix[i][j])
            vctr = np.array(lst)
            d.append(vctr)
        return d

    def cosine_similarity(self, a, b): #Sinartisi poy ypologizei omoiothta metaxi dyo dianismatwn gia thn eyresh ths omoiothtas gia dianysmata enos doc kai enos query
        dot_product = sum(a * b for a, b in zip(a, b))
        norm_a = sum(a * a for a in a) ** 0.5
        norm_b = sum(b * b for b in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot_product / (norm_a * norm_b)

    def vector_for_query(self, query): #Sinartisi poy dhmioyrgei to dianysma gia ena query ypologizontas opws kai sta docs ta tf-idf varh toy kathe oroy
        clean_query = self.clean_text(query)
        terms = clean_query.split()
        counter = collections.Counter(terms)
        query_vector = np.zeros(len(self.index) + 1)

        for i, word in enumerate(self.index.keys(), start=0): #keys() dinei ta words apo to query
            if word in terms:
                idf_list = self.index[word]
                ni = len(idf_list)
                idf = math.log2(1 + ((N - 1) / ni))
                f_ij = counter[word]
                tf = 1 + math.log2(f_ij)
                query_vector[i] = round(tf * idf, 3)
    
        return query_vector

#Anoigma arxeioy queries gia fortwsh twn erwthmatwn
file_name = 'Queries_20.txt'
queries = []
with open(file_name, 'r') as file:
    lines = file.readlines()

index = InvertedIndex()

#Index query kai apothikeysh se ena arxeio me thn morfh poy exoyn ta docs apo to opoio dhmiourgeite to katharo query gia ta vectors
for line in lines:
    words = line.split()
    query_file_name = 'query.txt'
    with open(query_file_name, 'w') as query_file:
        for word in words:
            query_file.write(word + "\n")

    with open(query_file_name, 'r') as query_file:
        cleaned_query = index.clean_text(query_file.read())
    
    queries.append(cleaned_query)

start1 = time.time()
#Dhmioyrgia eyrethrioy gia kathe doc
for doc_id in range(1, N):
    file_path = 'docs/' + f'{doc_id:05}' + '.txt'

    exists = os.path.exists(file_path)
    if exists:
        with open(file_path, 'r') as f:
            index.index_document(f.read(), doc_id)
end1 = time.time()

#Dhmioyrgia vectors gia kathe doc
matrix = index.vector_space(N)
vector_list = index.get_vectors(matrix, N)

#Dhmioyrgia twn vectors gia kathe query
query_vectors = []
for query in queries:
    vector = index.vector_for_query(query)
    query_vectors.append(vector)

#Dhmioyrgia enos dataframe gia ton elegxo kai ypologismo twn metrikwn
relevant_id = [] 
relevant_text = []
f = open('Relevant_20.txt', 'r')

text = f.readlines()

count = 1
for line in text:
  relevant_id.append(count)
  count += 1
  relevant_text.append(line.split()) #Split gia na lavei kathe stoixeio(dhladh kathe id) poy yparxei se kathe grammh toy relevant ws stoixeia listas kai oxi ws ena synoliko string

f.close()

relevant_data = {
  "relevant_id": relevant_id,
  "text": relevant_text
}

relevant_df = pd.DataFrame(relevant_data)

#Listes kai metavlhtes gia ypologismo metrikwn
Average_Precision = [] 
Reciprocal_Rank = []
num_queries = len(queries)
data =	{ #Ftiaxe ena dataframe gia to recall-precision diagramma opoy values oi times precision kai index oi times sta pososta gia to recall gia ola ta queries synolika
    "Precision": [0,0,0,0,0,0,0,0,0,0,0]
  }
Recall_Precision_df = pd.DataFrame(data, index = ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"])

#Eisodos toy xrhsth gia posa apotelesmata thelei sto telos sxetika me thn omoiothta kai tis metrikes
k = -1
while k < 0 or k > 1210:
    k = int(input(f"Give a number of top results by similarity for each query to be presented(From 0 to {N-1}, -1 to exit): "))
    if k == -1:
        print("Goodbye!")
        exit()

start2 = time.time()
#Ypologise to similarity gia kathe query me kathe doc kai emfanise ta kalytera k apotelesmata
for query_id,item in enumerate(query_vectors):
    sim = []
    rdoc_list = []
    for doc_id in range(1, N):
        document_vector = vector_list[doc_id]
        similarity = index.cosine_similarity(item, document_vector)*100 #Ypologismos similarity apo thn sinartisi cosine similarity kai metatroph se pososto
        sim.append([similarity,doc_id]) #Lista poy periexei th timh omoiothtas enos query me ena doc kai to id toy doc aytoy

    sort_sim = sorted(sim, reverse=True) #Sort th lista apo to megalytero similarity sto mikrotero

    print(f"\nTop {k} results for query {query_id+1}:") #Emfanise ta k apotelesmata gia kathe query
    for i in range(k):
        rdoc_list.append(sort_sim[i][1]) #Sth lista me ta sxetika keimena vale ta kalytera k apotelesmata
        print(f"\tSimilarity score: {sort_sim[i][0]:.4f} \tDocument id: {sort_sim[i][1]}")
    
    num_rel = len(relevant_df.loc[query_id]['text']) #Ypologismos twn relevant stoixeiwn gia ena query apo to dataframe me ta relevant keimena
    
    query_rp_df = pd.DataFrame(data, index = ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"]) #Dataframe gia to ekastote query ta apotelesmata toy opoioy tha prostethoyn sto geniko recall-precision dataframe
    
    count = 0 #Arithmos sxetikwn poy vriskontai stis apanthseis
    
    precision = [] #Lista gia to precision se kathe thesh poy yparxei sxetiko doc
    
    first = True #Metavlhth gia to prwto sxetiko doc gia ton ypologismo toy MRR
    
    for id in rdoc_list: #Gia kathe doc sth lista apanthsewn elegxe to me kathe doc sth lista sxetikwn
        for relevant in relevant_df.loc[query_id]['text']:
            if int(id) == int(relevant): #An to doc einai sxetiko me to query ayxhse to count
                count += 1
                position = rdoc_list.index(id) + 1 #Vres to position toy doc stis apanthseis (+1 kathos python xekina apo to 0)
                p = count/position #Ypologise to precision sth thesh ayth vasei twn sxetikwn (p@k)
                precision.append(p) #Prosthese to sth lista precision   
            
                if first: #An to first true tote to id einai to prwto sxetiko stis apanthseis ypologise to rr kai prosthese to sth lista enw kane to first false
                    rr = 1/position
                    Reciprocal_Rank.append(rr)
                    first = False
                
                recall = (count/num_rel)*100 #Vres to pososto anaklhshs analoga me to posa sxetika exoyn vrethei kathe fora

                if (0 <= recall) & (float(query_rp_df.loc["0%"]) == 0): #Gia kathe pososto mikrotero toy recall poy den exei oristei hdh orise thn timh toy ws to precision epi %
                    query_rp_df.loc["0%"] = p*100
                if (10 <= recall) & (float(query_rp_df.loc["10%"]) == 0):
                    query_rp_df.loc["10%"] = p*100
                if (20 <= recall) & (float(query_rp_df.loc["20%"]) == 0):
                    query_rp_df.loc["20%"] = p*100
                if (30 <= recall) & (float(query_rp_df.loc["30%"]) == 0):
                    query_rp_df.loc["30%"] = p*100
                if (40 <= recall) & (float(query_rp_df.loc["40%"]) == 0):
                    query_rp_df.loc["40%"] = p*100
                if (50 <= recall) & (float(query_rp_df.loc["50%"]) == 0):
                    query_rp_df.loc["50%"] = p*100
                if (60 <= recall) & (float(query_rp_df.loc["60%"]) == 0):
                    query_rp_df.loc["60%"] = p*100
                if (70 <= recall) & (float(query_rp_df.loc["70%"]) == 0):
                    query_rp_df.loc["70%"] = p*100
                if (80 <= recall) & (float(query_rp_df.loc["80%"]) == 0):
                    query_rp_df.loc["80%"] = p*100
                if (90 <= recall) & (float(query_rp_df.loc["90%"]) == 0):
                    query_rp_df.loc["90%"] = p*100
                if (100 == recall) & (float(query_rp_df.loc["100%"]) == 0):
                    query_rp_df.loc["100%"] = p*100
    
    Recall_Precision_df = Recall_Precision_df.add(query_rp_df) #Prosthese ta apotelesmata gia recall-precision toy query sto synoliko dataframe

    if first: #Se periptwsh poy stis apanthseis den yparxei sxetiko keimeno gia kapoio erwthma tote to RR einai 0
        Reciprocal_Rank.append(0)

    total = sum(precision) #Synolo twn epimeroys precision (An to precision einai keno tote total 0)

    if count != 0: #An yphrxan sxetika keimena stis apanthseis tote vres total einai to Average_Precision gia to query alliws einai 0
        total /= count
    Average_Precision.append(total)

map = sum(Average_Precision)/len(Average_Precision) #Ypologismos MAP kai MRR
mrr = sum(Reciprocal_Rank)/len(Reciprocal_Rank)

Recall_Precision_df = (Recall_Precision_df/num_queries).round(4) #Sto dataframe me to synoliko recall-precision gia ola ta queries diairese me ton arithmo twn queries gia na vreis tis meses times
ax = Recall_Precision_df.plot(title="Recall-Precision diagram", xlabel="Recall percentage", ylabel="Precision percentage", yticks = [0,10,20,30,40,50,60,70,80,90,100]) #Ftiakse to diagramma recall-precision me ta stoixeia toy dataframe
ax.set_xticks(range(len(Recall_Precision_df)))
ax.set_xticklabels(["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"])
ax.set_yticklabels(["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"])

print(f"\nThe MAP value over all 19 queries for k = {k} is equal to {map:.4f}") #Emfanish apotelesmatwn metrikwn MAP kai MRR
print(f"\nThe MRR value over all 19 queries for k = {k} is equal to {mrr:.4f}")
end2 = time.time()
plt.show() #Emfanise to diagramma recall-precision

total1 = end1 - start1 #Ypologismos xronoy eyrethriashs
total2 = end2 - start2 #Ypologismos xronoy paragwghs apotelesmatwn

print(f"\nTotal elapsed time for the index to be created: {total1:.4f} seconds")
print(f"\nTotal time elapsed to produce the results: {total2:.4f} seconds")
