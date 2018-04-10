#Import Packages

import urllib2,json
from bs4 import BeautifulSoup
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import socket
import httplib
import numpy as np
import math
import pymc3 as pm
import theano.tensor as T
from datetime import timedelta 
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

#Import Sklearn

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model

#Hyponetuse = Distnace of Shot

def distance(x1, y1,x2,y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

#GET All Dates and Game Information --- GET GAME IDS

request = urllib2.Request('https://statsapi.web.nhl.com/api/v1/schedule?startDate=2017-12-02&endDate=2018-01-30')
response = urllib2.urlopen(request)
json_load = json.loads(response.read())
#json_normalize(json_load['dates'])
games_dates = json_normalize(data=json_load['dates'], record_path=['games'], meta=['id'],errors='ignore') #SO FUCKING HELPUFL #https://www.kaggle.com/jboysen/quick-tutorial-flatten-nested-json-in-pandas
gamesids = games_dates[games_dates['gameType'] == 'R']['gamePk'] #Only Regular Season
gamesids.reset_index(inplace = True,drop = True)

#GET Plays Broken Out

for x in range(len(gamesids)):
    gameid = gamesids[x]
    url_start = 'https://statsapi.web.nhl.com/api/v1/game/'
    url_end = '/feed/live'
    full_url = url_start + str(gameid) + url_end
    request = urllib2.Request(full_url)
    response = urllib2.urlopen(request)
    json_load = json.loads(response.read())
    if 'Total_Plays' in globals():
        Plays = json_normalize(data=json_load['liveData']['plays']['allPlays'])
        Plays ['gameid'] = gameid
        frames = [Total_Plays,Plays]
        Total_Plays = pd.concat(frames)
    else:
        Total_Plays = json_normalize(data=json_load['liveData']['plays']['allPlays'])
        Total_Plays['gameid'] = gameid

#Previous Play --- You still have not controlled for a bunch of shit but it should be fine given when shots and goals occur

Total_Plays.sort_values(by = ['gameid','about.eventIdx'], axis=0, ascending=True,inplace = True)
Total_Plays['Previous_Play_Game'] = Total_Plays['gameid'].shift()
Total_Plays['Previous_Play'] = np.where(Total_Plays['Previous_Play_Game'] != Total_Plays['gameid'],np.nan,Total_Plays['result.event'].shift())

#Time Shifts 

Total_Plays['about.dateTime'] = Total_Plays['about.dateTime'].astype(str)
Total_Plays['about.dateTime'] = Total_Plays.apply(lambda x: x['about.dateTime'].replace('Z','').replace('T',' '),axis =1)
Total_Plays['about.dateTime'] = pd.to_datetime(Total_Plays['about.dateTime'])
Total_Plays['Previous_Play_Time'] = Total_Plays['about.dateTime'].shift()
#Total_Plays['Previous_Play_Time'] = Total_Plays['Previous_Play_Time'].astype(str)
#Total_Plays['Previous_Play_Time'] = Total_Plays.apply(lambda x: x['Previous_Play_Time'].replace('Z','').replace('T',' '),axis =1)
#Total_Plays['Previous_Play_Time'] = pd.to_datetime(Total_Plays['Previous_Play_Time'])
#Total_Plays['Time_Difference'] = Total_Plays['about.dateTime']-Total_Plays['Previous_Play_Time']
Total_Plays['Time_Difference'] = Total_Plays.apply(lambda r: (r['about.dateTime'] - r['Previous_Play_Time']).total_seconds(),axis = 1)

#Only Full Strength Moments -- STill not done

Total_Plays['Game_Time_Clock'] = (((Total_Plays['about.period'] - 1) * 20) * 60) + (Total_Plays['about.periodTime'].str.split(':').str[0].astype(float) * 60) + Total_Plays['about.periodTime'].str.split(':').str[1].astype(float)
Total_Plays['Penalty_Time'] = np.where(Total_Plays['result.event']  == 'Penalty',Total_Plays['Game_Time_Clock'] + (Total_Plays['result.penaltyMinutes'] * 60),0)
Penalty_Plays = Total_Plays[['gameid','Game_Time_Clock','Penalty_Time']][Total_Plays['Penalty_Time'] != 0]
Penalty_Plays.reset_index(inplace = True,drop = True)

#Define Plays
        
Shot_Plays_List = ['SHOT','GOAL']
Total_Plays['Shot_Play_B'] = np.where(Total_Plays['result.eventTypeId'].isin(Shot_Plays_List),1,0)
Total_Plays['Goal_B'] = np.where(Total_Plays['result.eventTypeId'] == 'GOAL',1,0) 
Shot_Plays = Total_Plays[Total_Plays['result.eventTypeId'].isin(Shot_Plays_List)]
Shot_Plays.reset_index(inplace = True, drop = True)

#Get Rid of Penalties

for i in Penalty_Plays.index:
    gameId_penalty = Penalty_Plays['gameid'][i]
    Penalty_Start = Penalty_Plays['Game_Time_Clock'][i]
    Penalty_End = Penalty_Plays['Penalty_Time'][i]
    Shot_Plays = Shot_Plays[~Shot_Plays.index.isin(Shot_Plays[(Shot_Plays['gameid'] == gameId_penalty) & (Shot_Plays['Game_Time_Clock'] >= Penalty_Start) & (Shot_Plays['Game_Time_Clock'] <= Penalty_End)].index)]
    
#Add the Booleans

Shot_Plays['Blocked_Boolean'] = np.where(Shot_Plays['result.eventTypeId'] == 'BLOCKED_SHOT',1,0)

#Flip the Signs --- Because it Needs to be on One Side

Shot_Plays['coordinates.y'] = np.where(Shot_Plays['coordinates.x'] < 0 ,Shot_Plays['coordinates.y'] * -1,Shot_Plays['coordinates.y'])
Shot_Plays['coordinates.x'] = np.where(Shot_Plays['coordinates.x'] < 0 ,Shot_Plays['coordinates.x'] * -1,Shot_Plays['coordinates.x'])
#Shot_Plays = Shot_Plays[Shot_Plays['coordinates.x'] != 89]
#Shot_Plays = Shot_Plays[Shot_Plays['coordinates.y'] != 0] #Get Rid of Things that just wont work. 
plt.plot(Shot_Plays['coordinates.x'],Shot_Plays['coordinates.y'],'ro')

#Where Did the Goals Happen

#plt.plot(Non_Goal_Shots['coordinates.x'],Non_Goal_Shots['coordinates.y'],'ro')
#plt.plot(Goals['coordinates.x'],Goals['coordinates.y'],'g^')
#Goals[['coordinates.x','coordinates.y']].head(5)

##################THE DATA

#Get THe Goal coordiantes

Goal1CoordinateX1 = 89
Goal1CoordinatePP = 3
Goal1CoordinateNP = -3
Goal1CoordinateCG = 0

#Distinaces

Shot_Plays['Horizontal_Distance'] = Shot_Plays.apply(lambda x: Goal1CoordinateX1 - x['coordinates.x'],axis =1)
Shot_Plays['Vertical_Distance'] = Shot_Plays.apply(lambda x: abs(x['coordinates.y']),axis =1)
Shot_Plays['Distance_To_Goal'] = Shot_Plays.apply(lambda x: math.hypot(x['Horizontal_Distance'],x['coordinates.y']),axis =1)
#Shot_Plays['Distance_To_Goal'] = Shot_Plays.apply(lambda x: math.hypot(Goal1CoordinateX1 - x['coordinates.x'],x['coordinates.y']),axis =1)
Shot_Plays['Shot_Distance_NP'] = Shot_Plays.apply(lambda x: distance(x['coordinates.x'],x['coordinates.y'],Goal1CoordinateX1,Goal1CoordinateNP),axis = 1)
Shot_Plays['Shot_Distance_PP'] = Shot_Plays.apply(lambda x: distance(x['coordinates.x'],x['coordinates.y'],Goal1CoordinateX1,Goal1CoordinatePP),axis = 1)
Shot_Plays[['Horizontal_Distance','coordinates.y','Distance_To_Goal']].head(10)

#Shot Angle

Shot_Plays['Shot_Angle'] = Shot_Plays.apply(lambda row: abs(math.degrees(math.asin(row['coordinates.y']/row['Distance_To_Goal']))) if row['Distance_To_Goal'] > 0 else 180,axis = 1)
#Shot_Plays['PP_Shot_Angle'] = Shot_Plays.apply(lambda row: abs(math.degrees(math.asin(row['coordinates.y']/row['Shot_Distance_PP']))) if row['Shot_Distance_PP'] > 0 else 180,axis = 1)
#Shot_Plays['NP_Shot_Angle'] = Shot_Plays.apply(lambda row: abs(math.degrees(math.asin(row['coordinates.y']/row['Shot_Distance_NP']))) if row['Shot_Distance_NP'] > 0 else 180,axis = 1)

#Blocked SHots

Blocked_Shots = Shot_Plays[['Blocked_Boolean','Shot_Angle','Vertical_Distance','Horizontal_Distance','Distance_To_Goal']]
Blocked_Shots = Blocked_Shots[~Blocked_Shots.isin([np.nan, np.inf, -np.inf]).any(1)]
Successful_Shots = Shot_Plays[['Goal_B','Previous_Play','result.secondaryType','Shot_Angle','Distance_To_Goal']][Shot_Plays['Blocked_Boolean'] == 0]

#Start Creating Hierachy

one = 'result.secondaryType'
one_names_df = Successful_Shots.groupby(one).all()
names_one  = list()
one_index = Successful_Shots.groupby(one).all().reset_index().reset_index()[['index',one]]
one_index.rename(columns={'index':'index_one'}, inplace=True)
one_indexes = one_index['index_one'].values #For Model
one_count = len(one_indexes) #For Model

#Index Two

two = 'Previous_Play'
two_names_df = Successful_Shots.groupby([one,two]).all()
names_two = list(two_names_df.index)
two_index = Successful_Shots.groupby([one,two]).all().reset_index().reset_index()[['index',one,two]]
two_index.rename(columns={'index':'index_two'},inplace = True)
two_indexes_df = pd.merge(one_index, two_index, how='inner', on= one)
two_indexes = two_indexes_df['index_one'].values #For Model
two_count = len(two_indexes) #For Model

Indexed_Successful_Shot_DF = pd.merge(Successful_Shots, two_indexes_df, how='inner', on=[one,two]).reset_index()

#Model Preminents

observed = Indexed_Successful_Shot_DF['Goal_B']
Shot_Type_Index = Indexed_Successful_Shot_DF['index_one']
N = len(np.unique(Indexed_Successful_Shot_DF['index_one']))

with pm.Model() as unpooled_model:

    # Independent parameters for each county
    a = pm.Normal('a', 0, sd=100, shape=N) #Intercept
    b = pm.Normal('b', 0, sd=100, shape=N) #Coefficient for Shot Type

    # Model error
    # Calculate predictions given values
    # for intercept and slope (Comment 4)
    yhat = pm.invlogit(a[Shot_Type_Index] + b[Shot_Type_Index] * Indexed_Successful_Shot_DF.Shot_Angle.values)
 
    # Make predictions fit reality
    y = pm.Binomial('y', n=np.ones(Indexed_Successful_Shot_DF.shape[0]), p=yhat, observed= observed)
    
    #Run It
    
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace_h = pm.sample(2000, step = step, start = start,njobs = 2)




#Example Part 2

with pm.Model() as multilevel_model:
 
    # Hyperiors for intercept (Comment 1)
    mu_a = pm.StudentT('mu_a', nu=3, mu=0., sd=1.0)
    sigma_a = pm.HalfNormal('sigma_a', sd=1.0)
    a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=n_countries)

    
    # Hyperpriors for slope
    mu_b = pm.StudentT('mu_b', nu=3, mu=0., sd=1.0)
    sigma_b = pm.HalfNormal('sigma_b', sd=1.0)
    b = pm.Normal('b', mu=mu_b, sd=sigma_b, shape=n_countries)


 
    # Make things grok-able (Comment 3)
    a_inv = pm.Deterministic('a_inv',T.exp(a)/(1 + T.exp(a)))
    fin = pm.Deterministic('fin',T.exp(a + b)/(1 + T.exp(a + b)))
 
    # Calculate predictions given values
    # for intercept and slope (Comment 4)


#Example Part 3

with multilevel_model:
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace_h = pm.sample(2000, step = step, start = start,njobs = 2)








##################The Models

#Build the Dta Set


y = Blocked_Shots['Blocked_Boolean']
x = Blocked_Shots.drop('Blocked_Boolean',axis =1)
y_train, y_test, x_train, x_test = model_selection.train_test_split(y,x,test_size = .33)
x_coef_names = x_train.columns

#




#Set Up the Model(s)

BRR = linear_model.BayesianRidge()
LR = linear_model.LogisticRegression()
EN = linear_model.ElasticNet()

#Logistic Regression Manual

alphas = np.logspace(-10,10,100)
c = alphas**-1
Log_Reg_Params = {'C': c,'penalty' : ['l1','l2']} ###RIDGE ONLY, but wiiiiidddddddeee range of Alpahs
gs_Log_Reg = model_selection.GridSearchCV(estimator = LR ,param_grid=Log_Reg_Params,cv =5,scoring = "roc_auc") #WHy 5? Whats the Point? Come up with an Answer?
gs_Log_Reg.fit(x_train,y_train)

#Lets Go 

gs_log_reg_scores = pd.DataFrame(gs_Log_Reg.cv_results_)
best_LR = 







#Run Test Into Best Model 


Shot_Plays[['result.description','result.event','result.secondaryType','Shot_Angle','coordinates.y','Horizontal_Distance','Distance_To_Goal']].head(5)     



np.where(Shot_Plays['Distance_To_Goal'] == 0.0,180.0,np.where(Shot_Plays['coordinates.y'] > 0.0,abs(math.degrees(math.asin((Shot_Plays['coordinates.y']/Shot_Plays['Distance_To_Goal']))),1.0))
         
    




Shot_Plays[['Horizontal_Distance','coordinates.y','Distance_To_Goal']].dtypes

[Shot_Plays['Distance_To_Goal'] == 0]
        
        ['Horizontal_Distance','coordinates.y','Distance_To_Goal']].head(10)


#Shot Angles -- Pretty Sure There is Something Wrong Here But We Can Put it Together Later

X_distances = Shot_Plays['Distance_To_Goal'].value_counts(sort = False)
X_distances = X_distances.reset_index()
X_distances.columns = ['distances','number']
X_distances.sort_values(by = 'distances')

G_distances = Shot_Plays[].value_counts(sort = False)
G_distances = G_distances.reset_index()
G_distances.columns = ['distances','number']
G_distances.sort_values(by = 'distances')

Shot_Plays['Shot_Angle'] = Shot_Plays.apply(lambda x: abs(math.degrees(math.asin(x['coordinates.y']/x['Distance_To_Goal']))),axis =1)
        
#Draw Again

plt.plot(Non_Goal_Shots['coordinates.x'],Non_Goal_Shots['coordinates.y'],'ro')
plt.plot(Goals['coordinates.x'],Goals['coordinates.y'],'g^')
plt.plot([Goal1CoordinateX1, Goal1CoordinateX1 ], [Goal1CoordinateNP, Goal1CoordinatePP], 'k-', lw=5)




#####Model

with pm.Model() as multilevel_model:
 
    # Hyperiors for intercept (Comment 1)
    mu_a = pm.StudentT('mu_a', nu=3, mu=0., sd=1.0)
    sigma_a = pm.HalfNormal('sigma_a', sd=1.0)
 
    # Hyperpriors for slope
    mu_b = pm.StudentT('mu_b', nu=3, mu=0., sd=1.0)
    sigma_b = pm.HalfNormal('sigma_b', sd=1.0)
 
    # Model the intercept (Comment 2)
    a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=n_countries)
 
    # Model the slope
    b = pm.Normal('b', mu=mu_b, sd=sigma_b, shape=n_countries)
 
    # Make things grok-able (Comment 3)
    a_inv = pm.Deterministic('a_inv',
                             T.exp(a)/(1 + T.exp(a)))
    fin = pm.Deterministic('fin',
                           T.exp(a + b)/(1 + T.exp(a + b)))
 
    # Calculate predictions given values
    # for intercept and slope (Comment 4)
    yhat = pm.invlogit(a[country_idx] +
                       b[country_idx] * df.treated.values)
 
    # Make predictions fit reality
    y = pm.Binomial('y', n=np.ones(df.shape[0]), p=yhat,
                    observed=df.success.values)






#The Example -- Just To See tHis Work -- https://dsaber.com/2016/08/27/analyze-your-experiment-with-a-multilevel-logistic-regression-using-pymc3%E2%80%8B/

def make_dataframe(n, s):
    df = pd.DataFrame({
        'success': [0] * (n * 4),
        'country': ['Canada'] * (2 * n) + ['China'] * (2 * n),
        'treated': [0] * n + [1] * n + [0] * n + [1] * n
    })
 
    for i, successes in zip([n, n*2, n*3, n*4], s):
        df.loc[i - n:i - n + successes - 1, 'success'] = 1
 
    return df
 
# n, ss = 200, [60, 100, 110, 120]
n, ss = 100, [30, 50, 55, 60]
 
df = make_dataframe(n, ss)

le = preprocessing.LabelEncoder()
country_idx = le.fit_transform(df['country'])
n_countries = len(set(country_idx))



pm.traceplot(trace_h)





