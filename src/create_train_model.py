import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Script to gather NBA data to train and test a neural network
# 1a. data-gathering
# 1b. Pre-processing
# 2. data manipulation, creating train / test samples
# 3. standardizing data, training Network
# 4. Testing model

def createModel():
    LABEL_COLUMN = "WINorLOSS"

    TEAM_DICT = {
             "MIL" : "Milwaukee Bucks",
             "MIA" : "Miami Heat",
             "BOS" : "Boston Celtics",
             "TOR" : "Toronto Raptors",
             "PHI" : "Philadelphia 76ers",
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

    # PRE-PROCESSING
    games2015 = pd.read_csv("../data/games/nba-games-14-15.txt",index_col=4)
    games2015 = games2015.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    games2016 = pd.read_csv("../data/games/nba-games-15-16.txt",index_col=4)
    games2016 = games2016.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    games2017 = pd.read_csv("../data/games/nba-games-16-17.txt",index_col=4)
    games2017 = games2017.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    games2018 = pd.read_csv("../data/games/nba-games-17-18.txt",index_col=4)
    games2018 = games2018.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    games2019 = pd.read_csv("../data/games/nba-games-18-19.txt",index_col=4)
    games2019 = games2019.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    games2020 = pd.read_csv("../data/games/nba-games-2019-oct-nov-dec.txt", index_col=4)
    games2020 = games2020.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    # Extracting NBA team season averages
    data2015 = pd.read_csv("../data/league-stats/team-per-game-14-15.txt",index_col=1)
    data2015 = data2015.drop(columns=['Rk','G', 'MP'])

    data2016 = pd.read_csv("../data/league-stats/team-per-game-15-16.txt",index_col=1)
    data2016 = data2016.drop(columns=['Rk','G', 'MP'])

    data2017 = pd.read_csv("../data/league-stats/team-per-game-16-17.txt",index_col=1)
    data2017 = data2017.drop(columns=['Rk','G', 'MP'])

    data2018 = pd.read_csv("../data/league-stats/team-per-game-17-18.txt",index_col=1)
    data2018 = data2018.drop(columns=['Rk','G', 'MP'])

    data2019 = pd.read_csv("../data/league-stats/team-per-game-18-19.txt",index_col=1)
    data2019 = data2019.drop(columns=['Rk','G', 'MP'])

    data2020 = pd.read_csv("../data/league-stats/team-per-game.txt", index_col=1)
    data2020 = data2020.drop(columns=['Rk', 'G', 'MP'])

    # Extracting NBA Season opponent season averages
    dataOppStats2015 = pd.read_csv("../data/league-stats/opponent-stats-14-15.txt", index_col=1)
    dataOppStats2015 = dataOppStats2015.drop(columns=['Rk', 'G', 'MP'])

    dataOppStats2016 = pd.read_csv("../data/league-stats/opponent-stats-15-16.txt", index_col=1)
    dataOppStats2016 = dataOppStats2016.drop(columns=['Rk', 'G', 'MP'])

    dataOppStats2017 = pd.read_csv("../data/league-stats/opponent-stats-16-17.txt", index_col=1)
    dataOppStats2017 = dataOppStats2017.drop(columns=['Rk', 'G', 'MP'])

    dataOppStats2018 = pd.read_csv("../data/league-stats/opponent-stats-17-18.txt", index_col=1)
    dataOppStats2018 = dataOppStats2018.drop(columns=['Rk', 'G', 'MP'])

    dataOppStats2019 = pd.read_csv("../data/league-stats/opponent-stats-18-19.txt", index_col=1)
    dataOppStats2019 = dataOppStats2019.drop(columns=['Rk', 'G', 'MP'])

    dataOppStats2020 = pd.read_csv("../data/league-stats/opponent-stats.txt", index_col=1)
    dataOppStats2020 = dataOppStats2020.drop(columns=['Rk', 'G', 'MP'])

    # Getting NBA "Misc Stats": see league stats for columns
    misc2015 = pd.read_csv("../data/league-stats/misc-team-14-15.txt", index_col=1)
    misc2015 = misc2015.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    misc2016 = pd.read_csv("../data/league-stats/misc-team-15-16.txt", index_col=1)
    misc2016 = misc2016.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    misc2017 = pd.read_csv("../data/league-stats/misc-team-16-17.txt", index_col=1)
    misc2017 = misc2017.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    misc2018 = pd.read_csv("../data/league-stats/misc-team-17-18.txt", index_col=1)
    misc2018 = misc2018.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    misc2019 = pd.read_csv("../data/league-stats/misc-team-18-19.txt", index_col=1)
    misc2019 = misc2019.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    misc2020 = pd.read_csv("../data/league-stats/misc-team.txt", index_col=1,skiprows=[0])
    misc2020 = misc2020.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    nba_games = [games2015,games2016,games2017,games2018,games2019,games2020]

    # setting the winner of the game
    # sets value "True" to A team win - changes later
    for s in nba_games:
        s["A_Win"] = s["PTS"] > s["PTS.1"]

    games2020 = games2020.drop(columns=["PTS","PTS.1"])
    games2015 = games2015.drop(columns=["PTS","PTS.1"])
    games2016 = games2016.drop(columns=["PTS","PTS.1"])
    games2017 = games2017.drop(columns=["PTS","PTS.1"])
    games2018 = games2018.drop(columns=["PTS","PTS.1"])
    games2019 = games2019.drop(columns=["PTS","PTS.1"])

    # all the games are listed as home games; creating some away games
    away_sets = [games2016,games2017,games2020]
    home_sets = [games2015,games2018,games2019]

    # setting home flags, flipping win column for home games
    for s in home_sets:
        s["A_Win"] = ~s["A_Win"]
        s["Home"] = 1

    # switches the team sides
    for s in away_sets:
        s["Home/Neutral"] = s.index
        s.index = s["Visitor/Neutral"]
        s["Home"] = 0

    # removing extraneous columns and ordering columns
    col = ["Date", "Home/Neutral", "A_Win", "Home"]
    games2016 = games2016.drop(columns=["Visitor/Neutral"])
    games2016 = games2016.reindex(columns=col)
    games2017 = games2017.drop(columns=["Visitor/Neutral"])
    games2017 = games2017.reindex(columns=col)
    games2020 = games2020.drop(columns=["Visitor/Neutral"])
    games2020 = games2020.reindex(columns=col)

    # Creating inputs to Neural Network

    # adding team stats and opponent stats to each game for each team
    d5 = games2020.join(data2020,lsuffix="_d3", rsuffix="_d4")
    d6 = d5.join(data2020, on='Home/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2020)
    d8 = d7.join(dataOppStats2020, on='Home/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d9 = d8.join(misc2020)
    stat2020 = d9.join(misc2020, on='Home/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')


    d5 = games2016.join(data2016,lsuffix="_d3",rsuffix="_d4")
    d6 = d5.join(data2016, on='Home/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2016)
    d9 = d7.join(dataOppStats2016, on='Home/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d10 = d9.join(misc2016)
    stat2016 = d10.join(misc2016, on='Home/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')

    d5 = games2015.join(data2015,lsuffix="_d3",rsuffix="_d4")
    d6 = d5.join(data2015, on='Visitor/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2015)
    d10 = d7.join(dataOppStats2015, on='Visitor/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d11 = d10.join(misc2015)
    stat2015 = d11.join(misc2015, on='Visitor/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')

    d5 = games2017.join(data2017,lsuffix="_d3",rsuffix="_d4")
    d6 = d5.join(data2017, on='Home/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2017)
    d11 = d7.join(dataOppStats2017, on='Home/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d12 = d11.join(misc2017)
    stat2017 = d12.join(misc2017, on='Home/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')

    d5 = games2018.join(data2018,lsuffix="_d3",rsuffix="_d4")
    d6 = d5.join(data2018, on='Visitor/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2018)
    d12 = d7.join(dataOppStats2018, on='Visitor/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d13 = d12.join(misc2018)
    stat2018 = d13.join(misc2018, on='Visitor/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')

    d5 = games2019.join(data2019,lsuffix="_d3",rsuffix="_d4")
    d6 = d5.join(data2019, on='Visitor/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2019)
    d13 = d7.join(dataOppStats2019, on='Visitor/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d14 = d13.join(misc2019)
    stat2019 = d14.join(misc2019, on='Visitor/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')

    gamesFile = open("data_logs/nba_games.txt","w")
    check = stat2020.columns
    game_data = pd.concat([stat2020,stat2015,stat2016,stat2017,stat2018,stat2019],sort = True)
    game_data = game_data.reindex(columns=check)

    # randomizing row order
    row,col=game_data.shape
    game_data = game_data.sample(frac=1) # randomizing rows

    # writing to the data log
    game_data = game_data.drop(columns=["Home/Neutral"])
    gamesFile.write(game_data.to_csv(index=True))
    game_data = game_data.drop(columns=["Date"])

    # splitting into training and testing samples
    split = int(row * .25)
    train_data = game_data.iloc[0:split] # 20% of total data pool
    test_data = game_data.iloc[split:row]

    # getting training labels and cleaning training data
    # flipping A_Win to represent 0 = A team win
    train_labels = ~train_data["A_Win"]
    train_labels = train_labels.to_numpy(dtype=bool)
    train_data = train_data.drop(columns=["A_Win"])

    # test labels and cleaning
    test_labels = ~test_data["A_Win"]
    test_labels = test_labels.to_numpy(dtype=bool)
    test_data = test_data.drop(columns=["A_Win"])

    # LABEL NOTES: 0 Reflects A team win, 1 reflects B team

    # saves the data and labels into txt logs
    test_file = open("data_logs/test_data.txt", "w")
    train_file = open("data_logs/train_data.txt", "w")

    test_file.write(test_data.to_csv(index=True))
    np.savetxt('data_logs/test_labels.txt', test_labels)
    train_file.write(train_data.to_csv(index=True))
    np.savetxt('data_logs/train_labels.txt', train_labels)

    # standardizing data
    x,y = train_data.shape
    w,z = test_data.shape

    for i in range(z):
        test_data.iloc[:,i] = test_data.iloc[:,i] / test_data.iloc[:,i].max()
    for i in range(y):
        train_data.iloc[:,i] = train_data.iloc[:,i] / train_data.iloc[:,i].max()

    row_t, col_t = train_data.shape

    # converting data into arrays
    train_data = train_data.values
    test_data = test_data.values

    # creating the model
    model = keras.Sequential([
        keras.layers.Input(col_t),
        keras.layers.Dense(64, activation='tanh'),
        keras.layers.Dense(16, activation='tanh'),
        keras.layers.Dense(2, activation='softmax')
    ])

    epochs = input('enter epochs: ')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, train_labels, epochs=int(epochs))
    model.evaluate(test_data,test_labels)

    # how many home wins predicted over test data
    predict = model.predict(test_data)
    count = 0
    for i in range(len(predict)):
        if(predict[i][0] > .5):
            count += 1
    print(count)

    np.savetxt('data_logs/test_predict.txt', predict)

    # results = open('data_logs/results.txt', "w")
    #
    # # writing which games were predicted incorrectly/correctly
    # wrong = 0
    # correct = 0
    # for i in range(len(predict)):
    #     a = np.argmax(predict[0])
    #     if a == test_labels[i]:
    #         results.write("WRONG\n")
    #         wrong += 1
    #     else:
    #         results.write("CORRECT\n")
    #         correct += 1
    #
    # res = str(correct) + " CORRECT | " + str(wrong) + " WRONG"
    # results.write(res)


    model.save("model/test_model.h5")


#creates model to predict point differential of an NBA game
def createDifferentialModel():
    LABEL_COLUMN = "WINorLOSS"

    TEAM_DICT = {
             "MIL" : "Milwaukee Bucks",
             "MIA" : "Miami Heat",
             "BOS" : "Boston Celtics",
             "TOR" : "Toronto Raptors",
             "PHI" : "Philadelphia 76ers",
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

    # PRE-PROCESSING
    games2015 = pd.read_csv("../data/games/nba-games-14-15.txt",index_col=4)
    games2015 = games2015.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    games2016 = pd.read_csv("../data/games/nba-games-15-16.txt",index_col=4)
    games2016 = games2016.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    games2017 = pd.read_csv("../data/games/nba-games-16-17.txt",index_col=4)
    games2017 = games2017.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    games2018 = pd.read_csv("../data/games/nba-games-17-18.txt",index_col=4)
    games2018 = games2018.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    games2019 = pd.read_csv("../data/games/nba-games-18-19.txt",index_col=4)
    games2019 = games2019.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    games2020 = pd.read_csv("../data/games/nba-games-2019-oct-nov-dec.txt", index_col=4)
    games2020 = games2020.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])

    # Extracting NBA team season averages
    data2015 = pd.read_csv("../data/league-stats/team-per-game-14-15.txt",index_col=1)
    data2015 = data2015.drop(columns=['Rk','G', 'MP'])

    data2016 = pd.read_csv("../data/league-stats/team-per-game-15-16.txt",index_col=1)
    data2016 = data2016.drop(columns=['Rk','G', 'MP'])

    data2017 = pd.read_csv("../data/league-stats/team-per-game-16-17.txt",index_col=1)
    data2017 = data2017.drop(columns=['Rk','G', 'MP'])

    data2018 = pd.read_csv("../data/league-stats/team-per-game-17-18.txt",index_col=1)
    data2018 = data2018.drop(columns=['Rk','G', 'MP'])

    data2019 = pd.read_csv("../data/league-stats/team-per-game-18-19.txt",index_col=1)
    data2019 = data2019.drop(columns=['Rk','G', 'MP'])

    data2020 = pd.read_csv("../data/league-stats/team-per-game.txt", index_col=1)
    data2020 = data2020.drop(columns=['Rk', 'G', 'MP'])

    # Extracting NBA Season opponent season averages
    dataOppStats2015 = pd.read_csv("../data/league-stats/opponent-stats-14-15.txt", index_col=1)
    dataOppStats2015 = dataOppStats2015.drop(columns=['Rk', 'G', 'MP'])

    dataOppStats2016 = pd.read_csv("../data/league-stats/opponent-stats-15-16.txt", index_col=1)
    dataOppStats2016 = dataOppStats2016.drop(columns=['Rk', 'G', 'MP'])

    dataOppStats2017 = pd.read_csv("../data/league-stats/opponent-stats-16-17.txt", index_col=1)
    dataOppStats2017 = dataOppStats2017.drop(columns=['Rk', 'G', 'MP'])

    dataOppStats2018 = pd.read_csv("../data/league-stats/opponent-stats-17-18.txt", index_col=1)
    dataOppStats2018 = dataOppStats2018.drop(columns=['Rk', 'G', 'MP'])

    dataOppStats2019 = pd.read_csv("../data/league-stats/opponent-stats-18-19.txt", index_col=1)
    dataOppStats2019 = dataOppStats2019.drop(columns=['Rk', 'G', 'MP'])

    dataOppStats2020 = pd.read_csv("../data/league-stats/opponent-stats.txt", index_col=1)
    dataOppStats2020 = dataOppStats2020.drop(columns=['Rk', 'G', 'MP'])

    # Getting NBA "Misc Stats": see league stats for columns
    misc2015 = pd.read_csv("../data/league-stats/misc-team-14-15.txt", index_col=1)
    misc2015 = misc2015.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    misc2016 = pd.read_csv("../data/league-stats/misc-team-15-16.txt", index_col=1)
    misc2016 = misc2016.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    misc2017 = pd.read_csv("../data/league-stats/misc-team-16-17.txt", index_col=1)
    misc2017 = misc2017.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    misc2018 = pd.read_csv("../data/league-stats/misc-team-17-18.txt", index_col=1)
    misc2018 = misc2018.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    misc2019 = pd.read_csv("../data/league-stats/misc-team-18-19.txt", index_col=1)
    misc2019 = misc2019.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    misc2020 = pd.read_csv("../data/league-stats/misc-team.txt", index_col=1,skiprows=[0])
    misc2020 = misc2020.drop(columns=['Rk', 'Arena','Attend.', 'Attend./G'])

    nba_games = [games2015,games2016,games2017,games2018,games2019,games2020]

    # setting the winner of the game
    for s in nba_games:
        s["DIFF"] = s["PTS"] - s["PTS.1"]

    games2020 = games2020.drop(columns=["PTS","PTS.1"])
    games2015 = games2015.drop(columns=["PTS","PTS.1"])
    games2016 = games2016.drop(columns=["PTS","PTS.1"])
    games2017 = games2017.drop(columns=["PTS","PTS.1"])
    games2018 = games2018.drop(columns=["PTS","PTS.1"])
    games2019 = games2019.drop(columns=["PTS","PTS.1"])

    # all the games are listed as home games; creating some away games
    away_sets = [games2016,games2017,games2020]
    home_sets = [games2015,games2018,games2019]

    # setting home flags, flipping win column for home games
    for s in home_sets:
        s["Home"] = 1

    # switches the team sides
    for s in away_sets:
        s["Home/Neutral"] = s.index
        s.index = s["Visitor/Neutral"]
        s["Home"] = 0

    # removing extraneous columns and ordering columns
    col = ["Date", "Home/Neutral", "DIFF", "Home"]
    games2016 = games2016.drop(columns=["Visitor/Neutral"])
    games2016 = games2016.reindex(columns=col)
    games2017 = games2017.drop(columns=["Visitor/Neutral"])
    games2017 = games2017.reindex(columns=col)
    games2020 = games2020.drop(columns=["Visitor/Neutral"])
    games2020 = games2020.reindex(columns=col)

    # Creating inputs to Neural Network

    # adding team stats and opponent stats to each game for each team
    d5 = games2020.join(data2020,lsuffix="_d3", rsuffix="_d4")
    d6 = d5.join(data2020, on='Home/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2020)
    d8 = d7.join(dataOppStats2020, on='Home/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d9 = d8.join(misc2020)
    stat2020 = d9.join(misc2020, on='Home/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')


    d5 = games2016.join(data2016,lsuffix="_d3",rsuffix="_d4")
    d6 = d5.join(data2016, on='Home/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2016)
    d9 = d7.join(dataOppStats2016, on='Home/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d10 = d9.join(misc2016)
    stat2016 = d10.join(misc2016, on='Home/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')

    d5 = games2015.join(data2015,lsuffix="_d3",rsuffix="_d4")
    d6 = d5.join(data2015, on='Visitor/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2015)
    d10 = d7.join(dataOppStats2015, on='Visitor/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d11 = d10.join(misc2015)
    stat2015 = d11.join(misc2015, on='Visitor/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')

    d5 = games2017.join(data2017,lsuffix="_d3",rsuffix="_d4")
    d6 = d5.join(data2017, on='Home/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2017)
    d11 = d7.join(dataOppStats2017, on='Home/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d12 = d11.join(misc2017)
    stat2017 = d12.join(misc2017, on='Home/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')

    d5 = games2018.join(data2018,lsuffix="_d3",rsuffix="_d4")
    d6 = d5.join(data2018, on='Visitor/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2018)
    d12 = d7.join(dataOppStats2018, on='Visitor/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d13 = d12.join(misc2018)
    stat2018 = d13.join(misc2018, on='Visitor/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')

    d5 = games2019.join(data2019,lsuffix="_d3",rsuffix="_d4")
    d6 = d5.join(data2019, on='Visitor/Neutral', lsuffix="_A",rsuffix="_B")
    d7 = d6.join(dataOppStats2019)
    d13 = d7.join(dataOppStats2019, on='Visitor/Neutral', lsuffix='_AOppStat', rsuffix='_BOppStats')
    d14 = d13.join(misc2019)
    stat2019 = d14.join(misc2019, on='Visitor/Neutral', lsuffix='_AMisc', rsuffix='_BMisc')

    gamesFile = open("data_logs/nba_games.txt","w")
    check = stat2020.columns
    game_data = pd.concat([stat2020,stat2015,stat2016,stat2017,stat2018,stat2019],sort = True)
    game_data = game_data.reindex(columns=check)

    gamesFile = open("data_logs/nba_games.txt","w")
    check = stat2020.columns
    game_data = pd.concat([stat2020,stat2015,stat2016,stat2017,stat2018,stat2019],sort = True)
    game_data = game_data.reindex(columns=check)

    # randomizing row order
    row,col=game_data.shape
    game_data = game_data.sample(frac=1) # randomizing rows

    # writing to the data log
    game_data = game_data.drop(columns=["Home/Neutral"])
    gamesFile.write(game_data.to_csv(index=True))
    game_data = game_data.drop(columns=["Date"])

    # splitting into training and testing samples
    split = int(row * .25)
    train_data = game_data.iloc[0:split] # 20% of total data pool
    test_data = game_data.iloc[split:row]

    # getting training labels and cleaning training data
    train_labels = train_data["DIFF"]
    train_labels = train_labels.to_numpy()
    train_data = train_data.drop(columns=["DIFF"])

    # test labels and cleaning
    test_labels = test_data["DIFF"]
    test_labels = test_labels.to_numpy()
    test_data = test_data.drop(columns=["DIFF"])

    # LABEL NOTES: 0 Reflects A team win, 1 reflects B team

    # saves the data and labels into txt logs
    test_file = open("data_logs/test_data.txt", "w")
    train_file = open("data_logs/train_data.txt", "w")

    test_file.write(test_data.to_csv(index=True))
    np.savetxt('data_logs/test_labels.txt', test_labels)
    train_file.write(train_data.to_csv(index=True))
    np.savetxt('data_logs/train_labels.txt', train_labels)

    # standardizing data
    x,y = train_data.shape
    w,z = test_data.shape

    for i in range(z):
        test_data.iloc[:,i] = test_data.iloc[:,i] / test_data.iloc[:,i].max()
    for i in range(y):
        train_data.iloc[:,i] = train_data.iloc[:,i] / train_data.iloc[:,i].max()

    row_t, col_t = train_data.shape

    # converting data into arrays
    train_data = train_data.values
    test_data = test_data.values

    # creating the model
    model = keras.Sequential([
        keras.layers.Input(col_t),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='elu')
    ])

    epochs = input('enter epochs: ')
    model.compile(optimizer="adam", loss='mean_squared_error', metrics=["accuracy"])
    model.fit(train_data, train_labels, epochs=int(epochs))
    model.evaluate(test_data,test_labels)

    # how many home wins predicted over test data
    predict = model.predict(test_data)
    count = 0
    for i in range(len(predict)):
        if predict[i] < 0:
            if predict[i] > test_labels[i]:
                count+=1
        elif predict[i] > 0:
            if predict[i] < test_labels[i]:
                count+=1
    print(count/len(predict))

    np.savetxt('data_logs/test_predict.txt', predict)

    model.save("model/test_model.h5")
