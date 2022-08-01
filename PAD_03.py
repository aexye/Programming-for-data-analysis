import pandas as pd

#Załadowanie pliku
filename = r'P:/b.balcerzak/2020/INFORMATYKA_ZAOCZNE/SEM_1/PAD/PAD_03/Zadanie_domowe/bank.csv'
df = pd.read_csv(filename, sep=';')

#Zadanie1
#podpunkt a 

phone_count = df.groupby(by='month')['contact'].count().idxmax()
print('Month with highest count of phone calls was: {}'.format(phone_count))

#podpunkt b

avg_duration = df.groupby(by='month')['duration'].mean().idxmax()
print('Month with highest average of call duration was: {}'.format(avg_duration))

#podpunkt c

educ_savings = df.groupby(by='education')['balance'].mean().sort_values(ascending=False)
print(educ_savings)

#podpunkt d

oldest = df.groupby(by='marital')['age'].max()

oldest_married = oldest.filter(like='married').array
oldest_single = oldest.filter(like='single').array

print('Oldest married person in dataset is {} years old. Oldest single person in dataset is {} years old.'.format(oldest_married[0], oldest_single[0]))