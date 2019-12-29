# Brandon Chan
import pandas as pd
import numpy as np
# teamName: team name as a string
# teamName = "Milwaukee Bucks"

team_data = pd.read_csv("../data/league-stats/team-per-game.txt", index_col=1)
team_data = team_data.drop(columns=['Rk', 'G', 'MP'])

dataOppStats2020 = pd.read_csv("../data/league-stats/opponent-stats.txt", index_col=1)
dataOppStats2020 = dataOppStats2020.drop(columns=['Rk', 'G', 'MP'])

# a helper function that pre-processes data for input to the neural network
def getNeuralInputs(home,away):
    home_stats = team_data.loc[home].values
    away_stats = team_data.loc[away].values
    home_opp = dataOppStats2020.loc[home].values
    away_opp = dataOppStats2020.loc[away].values

    inputs = [[1,*home_stats,*away_stats,*home_opp,*away_opp]]

    return  inputs
