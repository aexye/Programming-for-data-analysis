import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Setting path for file
filename = r'P:/INFORMATYKA_ZAOCZNE/SEM_1/PAD/PAD_04/Zadanie_domowe/london_merged.csv'
#Reading file and saving it to dataframe
model = pd.read_csv(filename, sep=',')
model['timestamp'] = pd.to_datetime(model.timestamp, format='%Y-%m-%d %H:%M:%S')
#Deleting first column from analysis
model_r = model.drop(columns = 'timestamp')
#Normalization of data set
model_r = (model_r-model_r.mean())/model_r.std()

#abc
#Setting x, y for linear regression
y = model_r['cnt']
x = model_r.drop(columns ='cnt')

#Linear reggresion
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25) 
model_reg = LinearRegression()
model_reg.fit(x_train,y_train)

#Extracting hour and year from timestamp column
model2 = model_r.copy()
model2['hour'] = model['timestamp'].dt.hour
model2['year'] = model['timestamp'].dt.year

#Checking statistical significance of variables 
x = sm.add_constant(x_train)
model_stat = sm.OLS(y_train,x)
results = model_stat.fit()
results.params
p = results.pvalues
print(p[p<=0.05])

#Drawing a matrix
corrMatrix = model_r.corr()
plt.figure(figsize=(11,11))
plt.matshow(model_r.corr(),fignum=1)
plt.xticks(ticks = list(range(9)),labels = model_r.keys())
plt.yticks(ticks = list(range(9)),labels = model_r.keys())
plt.show()

#Creating second model based on year

unique_year = list(set(model2['year']))

#Creating a dataframe dictionary to store dataframes
DataFrameDict = {elem : pd.DataFrame for elem in unique_year}

for year in DataFrameDict.keys():
    DataFrameDict[year] = model2[:][model2.year == year]



#Regression analysis for each year

for year_model, df in DataFrameDict.items():
    #Indicator of the year
    print('\n\n')
    print(year_model)
    #Dropping unecessary columns
    df = df.drop(columns=['hour', 'year'])
    #Setting x, y for linear regression
    y = df['cnt']
    x = df.drop(columns ='cnt')
    #Linear reggresion
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25) 
    model_reg = LinearRegression()
    model_reg.fit(x_train,y_train)

    #Checking statistical significance of variables 
    x = sm.add_constant(x_train)
    model_stat = sm.OLS(y_train,x)
    results = model_stat.fit()
    results.params
    p = results.pvalues
    print(p[p<=0.05])

    #Drawing a matrix
    corrMatrix = df.corr()
    plt.figure(figsize=(11,11))
    plt.matshow(df.corr(),fignum=1)
    plt.xticks(ticks = list(range(9)),labels = df.keys())
    plt.yticks(ticks = list(range(9)),labels = df.keys())
    plt.show()
