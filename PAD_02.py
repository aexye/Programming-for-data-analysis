import pandas as pd

#Import file from folder
filename = r"D:/PAD_02_PD.csv"
df = pd.read_csv(filename, delimiter=';')

#Create total goods column based on owns * columns
df['Goods_total'] = df[['owns_car', 'owns_TV', 'owns_house', 'owns_Phone']].sum(axis=1)

#Calculate average goods for men and women
men = df[df['gender'] == 'M']
women = df[df['gender'] == 'K']

men_avg = men['Goods_total'].sum()/men.shape[0]
women_avg = women['Goods_total'].sum()/women.shape[0]

#m_f_avg = df.groupby('gender')['Goods_total'].mean()

#Create table grouped by 'Country' and giving statistics of it
new_df = df.groupby('Country').agg({'Goods_total':'mean',
                                    'Age':'min',
                                    'gender': lambda x: (x[x=='K'].count()/x.count())*100
                                    })

new_df = new_df.rename(columns={'Goods_total': 'avg_goods', 'Age': 'min_age', 'gender': 'women_avg_perc'})

#Check
print(new_df)
print(men_avg)
print(women_avg)






