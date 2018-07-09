#%matplotlib inline

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from nltk.tokenize import RegexpTokenizer


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import recall_score
sqlobject = sqlite3.connect('amazon-fine-foods/database.sqlite')

reviews = pd.read_sql_query("""SELECT Score, Summary FROM Reviews""", sqlobject)[:100000]

original = reviews.copy()
reviews = original.copy()

print reviews.shape

scores = reviews['Score']
reviews['Score'] = reviews['Score'].apply(lambda x : 'pos' if x > 3 else 'neg')

def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['Score']=='neg']
    pos = reviews.loc[Summaries['Score']=='pos']
    return [pos,neg]

[pos,neg] = splitPosNeg(reviews)


#stemmer = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
stop = stopwords.words('english')
translation = string.maketrans(string.punctuation,' '*len(string.punctuation))
tokenizer = RegexpTokenizer(r'\w+')
def preprocessing(line):
    tokens=[]
    line = tokenizer.tokenize(line.lower())
    # line = line.translate(translation)
    # line = nltk.word_tokenize(line.lower())
    for t in line:
        #if(t not in stop):
            #stemmed = stemmer.stem(t)
        stemmed = lemmatizer.lemmatize(t)
        tokens.append(stemmed)
    return ' '.join(tokens)

pos_data = []
neg_data = []
for p in pos['Summary']:
    pos_data.append(preprocessing(p))

for n in neg['Summary']:
    neg_data.append(preprocessing(n))
data = pos_data + neg_data
labels = np.concatenate((pos['Score'].values,neg['Score'].values))


[Data_train,Data_test,Train_labels,Test_labels] = train_test_split(data,labels , test_size=0.25, random_state=20160121)#,stratify=labels)

t_one = []
for line in Data_train:
    l = tokenizer.tokenize(line.lower())
    for w in l:
        t_one.append(w)

t_two = []
for line in Data_train:
    l = tokenizer.tokenize(line.lower())
    for i in range(len(l)-1):
        t_two.append(" ".join(l[i:i+2]))

t_three = []
for line in Data_train:
    l = tokenizer.tokenize(line.lower())
    for i in range(len(l)-2):
        t_three.append(" ".join(l[i:i+3]))

top_words = []
def print_word(destr,n_featr):
    word_features = nltk.FreqDist(destr)
    print len(word_features)
    for fpair in list(word_features.most_common(n_featr)):
        top_words.append(fpair[0])

print_word(t_one,5000)
print t_one[:50]
print t_two[:50]
print_word(t_two,1000)
print_word(t_three,500)

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(top_words)])

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)
print tf_fit
ctr_features = vec.transform(Data_train)
tr_features = tf_vec.transform(ctr_features)

cte_features = vec.transform(Data_test)
te_features = tf_vec.transform(cte_features)

models = {'Logistic' : linear_model.LogisticRegression(C=1e5),'Decision Tree' : DecisionTreeClassifier(random_state=20160121, criterion='entropy'),
          'Perceptron': linear_model.Perceptron(n_iter=1000)}

results = pd.DataFrame()

foldnum = 0
tfprediction = {}
cprediction = {}
for name,model in models.items():
        print name
        model.fit(tr_features, Train_labels)
        tfprediction[name] = model.predict(te_features)
        tfaccuracy = metrics.accuracy_score(tfprediction[name],Test_labels)
        print tfaccuracy
        #model.fit(ctr_features,Train_labels)
        #cprediction[name] = model.predict(cte_features)
        #caccuracy = metrics.accuracy_score(cprediction[name],Test_labels)

        results.loc[foldnum,'TF-IDF Accuracy']=tfaccuracy
        #results.loc[foldnum,'Count Accuracy']=caccuracy
        results.loc[foldnum,'Model']=name
        foldnum = foldnum+1
print (results)

for name,model in models.items():
    print ("Classification report for ",name)
    print(metrics.classification_report(Test_labels, tfprediction[name]))
    print("\n")
