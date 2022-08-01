from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics import *


def NounCount(x):
    nounCount = sum(1 for word, pos in x if pos.startswith('NN'))
    return nounCount
def AdjCount(x):
    AdjCount = sum(1 for word, pos in x if pos.startswith('JJ'))
    return AdjCount
def VerbCount(x):
    verbCount = sum(1 for word, pos in x if pos.startswith('VBP'))
    return verbCount

path = r'C:\Users\mostrowski194\Downloads\SMSSpamCollection.txt'

df = pd.read_csv(path, names =['0/1', 'text'], delimiter='\t')
df = pd.get_dummies(df, columns=['0/1'])
df.drop('0/1_ham', axis=1, inplace=True)
df['text'] = df.text.str.replace('[^a-zA-Z ]', '')
df['txt'] = df.text.apply(nltk.word_tokenize)
df['txt'] = df.txt.apply(nltk.pos_tag)
df['NOUN'] = df['txt'].apply(NounCount)/df['txt'].str.len()
df['VERB'] = df['txt'].apply(VerbCount)/df['txt'].str.len()
df['ADJ'] = df['txt'].apply(AdjCount)/df['txt'].str.len()
df.drop('txt', axis=1, inplace=True)

print(df.head())

tfidf = TfidfVectorizer(min_df=10)
tfidf_data = tfidf.fit_transform(df['text'])
target = df['0/1_spam']

train_input,test_input,train_target,test_target = train_test_split(tfidf_data, target, test_size=0.25)


#Logistic regression

model1 = LogisticRegression()
model1.fit(train_input,train_target)

#Linear regression

model2 = KMeans(n_clusters = 2).fit(tfidf_data)


#MultinomialNB

model3 = MultinomialNB()
model3.fit(train_input,train_target)

pred_logr = model1.predict(test_input)
pred_kmeans = model2.predict(test_input)
pred_nb = model3.predict(test_input)

acc_logr = round(accuracy_score(test_target,pred_logr),2)
acc_kmeans = round(accuracy_score(test_target,pred_kmeans),2)
acc_nb = round(accuracy_score(test_target,pred_nb),2)

print(acc_logr, acc_nb, acc_kmeans)