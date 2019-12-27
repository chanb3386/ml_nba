import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

LABELS = [True, False]
# True = home win, False = away win
LABEL_COLUMN = "WINorLOSS"

TEAM_DICT = {
             "MIL" : "Milwaukee Bucks",
             "MIA" : "Miami Heat",
             "BOS" : "Boston Celtics",
             "TOR" : "Toronto Raptors",
             "PHI" : "Philidelphia 76ers",
             "IND" : "Indiana Pacers",
             "BRK" : "Brooklyn Nets",
             "ORL" : "Orlando Magic",
             "CHO" : "Charlotte Hornets",
             "CHI" : "Chicago Bulls",
             "DET" : "Detroit Pistons",
             "WAS" : "Washington Wizards",
             "CLE" : "Cleveland Cavaliers",
             "NYK" : "New York Knicks",
             "ATL" : "Atlanta Hawks",
             "LAL" : "Los Angeles Lakers",
             "DEN" : "Denver Nuggets",
             "HOU" : "Houston Rockets",
             "LAC" : "Los Angeles Clippers",
             "DAL" : "Dallas Mavericks",
             "UTA" : "Utah Jazz",
             "OKC" : "Oklahoma City Thunder",
             "POR" : "Portland Trail Blazers",
             "SAS" : "San Antonio Spurs",
             "SAC" : "Sacramento Kings",
             "PHO" : "Phoenix Suns",
             "MEM" : "Memphis Grizzlies",
             "MIN" : "Minnesota Timberwolves",
             "NOP" : "New Orleans Pelicans",
             "GSW" : "Golden State Warriors"
            }

BOOL_DICT = {1 : True, 0 : False}


# FOR LATER USE - 2014-2018 data
data2014_2018 = pd.read_csv("../data/games/nba-games-results-2014-2018.csv",index_col=1)
data2014_2018 = data2014_2018.drop(columns=['Unnamed: 0','Game'])
data2014_2018['Home'] = pd.Categorical(data2014_2018['Home'])
data2014_2018['Home'] = data2014_2018.Home.cat.codes
data2014_2018['WINorLOSS'] = pd.Categorical(data2014_2018['WINorLOSS'])
data2014_2018['WINorLOSS'] = data2014_2018.WINorLOSS.cat.codes
data2014_2018.index = data2014_2018.index.map(TEAM_DICT)
data2014_2018['Opponent'] = data2014_2018['Opponent'].map(TEAM_DICT)
#

# Extracting NBA game results
data19_20 = pd.read_csv("../data/games/nba-games-2019-oct-nov-dec.txt", index_col=4)
data19_20 = data19_20.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

# Extracting NBA team season averages
data_team_stat = pd.read_csv("../data/league-stats/team-per-game.txt", index_col=1)
data_team_stat = data_team_stat.drop(columns=['Rk', 'G', 'MP'])

# FOR LATER USE
data2015 = pd.read_csv("../data/league-stats/team-per-game-14-15.txt",index_col=1)
data2015 = data2015.drop(columns=['Rk','G', 'MP'])

data2016 = pd.read_csv("../data/league-stats/team-per-game-15-16.txt",index_col=1)
data2016 = data2016.drop(columns=['Rk','G', 'MP'])

data2017 = pd.read_csv("../data/league-stats/team-per-game-16-17.txt",index_col=1)
data2017 = data2017.drop(columns=['Rk','G', 'MP'])

data2018 = pd.read_csv("../data/league-stats/team-per-game-17-18.txt",index_col=1)
data2018 = data2018.drop(columns=['Rk','G', 'MP'])
#

# Extracting NBA Season opponent season averages
dataOppStats2020 = pd.read_csv("../data/league-stats/opponent-stats.txt", index_col=1)
dataOppStats2020 = dataOppStats2020.drop(columns=['Rk', 'G', 'MP'])

#x,y = data_team_stat.shape

# standardizing data
data19_20["HomeWin"] = data19_20["PTS"] > data19_20["PTS.1"] # converts win/lose to a boolean for labels
data19_20 = data19_20.drop(columns=["PTS", "PTS.1"])

#w,z = data2014_2018.shape

data2014_2018 = data2014_2018[data2014_2018["Home"] == 1]
data2014_2018 = data2014_2018.drop(columns=['Home'])
data2014_2018 = data2014_2018.rename(columns = {"Opponent" : "Visitor/Neutral",
                                "WINorLOSS" : "HomeWin"})
data2014_2018 = data2014_2018.drop(columns = data2014_2018.columns[3:])
data2014_2018["HomeWin"] = data2014_2018["HomeWin"].map(BOOL_DICT)

# Creating inputs to Neural Network
d5 = data19_20.join(data_team_stat) # adds home team stats
d6 = d5.join(data_team_stat, on='Visitor/Neutral',lsuffix='_home',rsuffix='_away') # adds away team dataOppStats2020
d7 = d6.join(dataOppStats2020) # adds home opponent season averages
game_data = d7.join(dataOppStats2020, on='Visitor/Neutral',lsuffix='_homeOpp',rsuffix='_awayOpp') # adds away opp season averages
game_data = game_data.drop(columns=['Date'])

# randomizing rows so test, train samples from larger sample of teams
row,col=game_data.shape
game_data = game_data.sample(frac=1) # randomizing rows
split = int(row * .2)
train_data = game_data.iloc[0:split] # 20% of total data pool
test_data = game_data.iloc[split:row]

# getting training labels and cleaning training data
train_labels = train_data["HomeWin"]
train_labels = train_labels.to_numpy(dtype=bool)
train_data = train_data.drop(columns=["HomeWin"])

# test labels and cleaning
test_labels = test_data["HomeWin"]
test_labels = test_labels.to_numpy(dtype=bool)
test_data = test_data.drop(columns=["HomeWin"])

test_data = test_data.drop(columns=['Visitor/Neutral'])
train_data = train_data.drop(columns=['Visitor/Neutral'])

row_t, col_t = train_data.shape

# converting data into arrays
train_data = train_data.values
test_data = test_data.values

# creating the model
model = keras.Sequential([
    keras.layers.Input(col_t),
    keras.layers.Dense(256, activation='tanh'),
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_data, train_labels, epochs=5)
model.evaluate(test_data,test_labels)

model.save("predict_nba.h5")
