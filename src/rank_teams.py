import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import opendata
import predict

# ranks teams based on a simulated tournament:
# each team plays each other team twice, once at home, once away
# ranks teams on win % (58 games each team)
def rankTeams(model):
        homeTeams = ["Milwaukee Bucks","Miami Heat","Boston Celtics","Toronto Raptors","Philadelphia 76ers","Indiana Pacers","Brooklyn Nets",
                     "Orlando Magic","Charlotte Hornets","Chicago Bulls","Detroit Pistons","Washington Wizards","Cleveland Cavaliers","New York Knicks",
                     "Atlanta Hawks","Los Angeles Lakers","Denver Nuggets","Houston Rockets","Los Angeles Clippers","Dallas Mavericks","Utah Jazz",
                     "Oklahoma City Thunder","Portland Trail Blazers","San Antonio Spurs","Sacramento Kings","Phoenix Suns","Memphis Grizzlies",
                     "Minnesota Timberwolves","New Orleans Pelicans","Golden State Warriors"]

        awayTeams = ["Milwaukee Bucks","Miami Heat","Boston Celtics","Toronto Raptors","Philadelphia 76ers","Indiana Pacers","Brooklyn Nets",
                     "Orlando Magic","Charlotte Hornets","Chicago Bulls","Detroit Pistons","Washington Wizards","Cleveland Cavaliers","New York Knicks",
                     "Atlanta Hawks","Los Angeles Lakers","Denver Nuggets","Houston Rockets","Los Angeles Clippers","Dallas Mavericks","Utah Jazz",
                     "Oklahoma City Thunder","Portland Trail Blazers","San Antonio Spurs","Sacramento Kings","Phoenix Suns","Memphis Grizzlies",
                     "Minnesota Timberwolves","New Orleans Pelicans","Golden State Warriors"]

        # keeps track of a teams wins
        wins = {'Milwaukee Bucks':0, 'Miami Heat':0, "Boston Celtics":0,"Toronto Raptors":0,"Philadelphia 76ers":0,"Indiana Pacers":0,"Brooklyn Nets":0,
                     "Orlando Magic":0,"Charlotte Hornets":0,"Chicago Bulls":0,"Detroit Pistons":0,"Washington Wizards":0,"Cleveland Cavaliers":0,"New York Knicks":0,
                     "Atlanta Hawks":0,"Los Angeles Lakers":0,"Denver Nuggets":0,"Houston Rockets":0,"Los Angeles Clippers":0,"Dallas Mavericks":0,"Utah Jazz":0,
                     "Oklahoma City Thunder":0,"Portland Trail Blazers":0,"San Antonio Spurs":0,"Sacramento Kings":0,"Phoenix Suns":0,"Memphis Grizzlies":0,
                     "Minnesota Timberwolves":0,"New Orleans Pelicans":0,"Golden State Warriors":0}

        count = 0
        # matching teams up
        for i in range(len(homeTeams)):
            for j in range(len(awayTeams)):
                if homeTeams[i] != awayTeams[j]:
                    res = predict.predictNetwork2(homeTeams[i], awayTeams[j], model)
                    if(res[0][0] > .53):
                        winner = 0
                    elif(res[0][0] < .47):
                        winner = 1
                    else:
                        winner = 2

                    #winner = np.argmax(res[0])
                    if winner == 0:
                        wins[homeTeams[i]] += 1
                    elif winner == 1:
                        wins[awayTeams[j]] += 1
                    else:
                        print("between: " + str(count))
                        count+=1

        # printing results
        for k,v in wins.items():
            print(k+" : "+str(v))
        print("between count: " + str(count))
