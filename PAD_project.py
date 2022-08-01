import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

path = r'C:\Users\ostri\OneDrive\Pulpit\BostonHousing.csv'
df = pd.read_csv(path)

#Dropping chas column following tasks
df = df.drop('chas', axis=1)
#Setting x and y for reggresion analysis
y = df['medv']
x = df.drop('medv', axis=1)

#Standardization of data
x = (x - x.mean())/x.std()

#Splitting data into training and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(5,5))
plt.matshow(df.corr(),fignum=1)
plt.xticks(ticks = list(range(13)),labels = df.keys())
plt.yticks(ticks = list(range(13)),labels = df.keys())
#Graphic form
plt.show()
#Dataframe
print(corr_matrix)

#Modelling
#Linear regression

model_reg = LinearRegression()
model_reg.fit(x_train,y_train)

x = sm.add_constant(x_train)
model_stat = sm.OLS(y_train,x)
results = model_stat.fit()
results.params
y_train_predict = model_reg.predict(x_train)
r2_linreg = r2_score(y_train, y_train_predict)
p = results.pvalues
print('Parametrs that are statistically significant:\n{}'.format(p[p<=0.05]))

#KERAS model training

model = Sequential()
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='rmsprop', loss='mse')

model.fit(x_train,y_train, epochs=400)

y_train_predict2 = model.predict(x_train)
r2_keras = r2_score(y_train, y_train_predict2)

print('Standard linear model accuracy: {}, Keras trained model accuracy: {}'.format(r2_linreg, r2_keras))

#Scatterplot between medv and crim with trendline

scatter = sns.regplot(x=df['medv'], y=df['crim'])
plt.show()

#Dashboard with use of plotly and dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Creating bar chart including linear model params
fig = px.bar(p, log_y=True)

#App initialize
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#App
app.layout = html.Div([
    html.H1('DASHBOARD'),
    html.Div('Wykres słupkowy dla paramterów z modelu liniowego', style = {'color':'blue', 'font-size':'2em'}),
    html.Div([
    dcc.Graph(figure=fig)]),
    html.Div([
    html.Div('Histogram dla wybranej kolumny.', style = {'color':'blue', 'font-size':'2em'}),
    dcc.Dropdown(
        id="dropdown",
        options=list(df.columns),
        value=None,
        clearable=False,
    ),
    dcc.Graph(id="graph")]),
    html.Div([
    html.Div('Wykres rozrzutu dla wszystkich par kolumn.', style = {'color':'blue', 'font-size':'2em'}),
        html.Div([
            dcc.Dropdown(
                id="dropdown1",
                options=list(df.columns),
                value=None,
                clearable=False,
            )], style={'width': '49%', 'display': 'inline-block'}),
            html.Div([
                dcc.Dropdown(
                id="dropdown2",
                options=list(df.columns),
                value=None,
                clearable=False,
            )], style={'width': '49%', 'display': 'inline-block'})]),
    dcc.Graph(id="scatter-plot")
    ], style = { 'width': '80%', 'margin': '0 auto'})

#Dynamic elements like dropdown list to dynamically change content
@app.callback(
    Output("graph", "figure"), 
    [Input("dropdown", "value")])
def display_color(col):
    data = df[col]
    fig = px.histogram(data, nbins=30)
    return fig
@app.callback(
    Output("scatter-plot", "figure"), 
    [Input('dropdown1', 'value'),
     Input('dropdown2', 'value')])
def update_bar_chart(col1, col2):
    data = df
    fig = px.scatter(x=data[col1], y=data[col2])
    fig.update_xaxes(title=col1, type='linear')
    fig.update_yaxes(title=col2, type='linear')
    return fig
app.run_server(debug=True)
