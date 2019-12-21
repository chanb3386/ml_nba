# Brandon Chan
import pandas as pd
# teamName: team name as a string
# teamName = "Milwaukee Bucks"
def openTeamPlayerStats(teamName):
    teamNameArr = teamName.split()
    filepath = "../data/team-player-stats/" + teamNameArr[-1].lower() + ".txt"
    table = pd.read_csv(filepath, index_col=1)
    return table

# teamName: team name as a string or "League Average"
def openTeamLeagueStats(teamName):
    filepath = "..data/league-stats/team-per-game.txt"
    table = pd.read_csv(filepath, index_col=1)
    return table.loc[teamName]

def openAllLeagueStats():
    filepath = "..data/league-stats/team-per-game.txt"
    table = pd.read_csv(filepath, index_col=1)
    return table
