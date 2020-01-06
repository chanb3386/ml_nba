from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


# function that updates games, team stats and opp stats files for most recent season
# WIP - games, league stats, opp stats, misc stats
def updateData():
    url = "https://www.basketball-reference.com/leagues/NBA_2020.html"
    url_games = "https://www.basketball-reference.com/leagues/NBA_2020_games.html"

    driver = webdriver.Chrome('../cdriver/chromedriver')
    driver.get(url)

    time.sleep(2)

    actions = ActionChains(driver)

    teamStats = driver.find_element_by_tag_name('h2') # need to find element better
    actions.move_to_element(teamStats)
    actions.perform()
    time.sleep(2)
    #teamButton = driver.find_element_by_link_text('Share & More')
    #actions.move_to_element(teamButton)
    #print(teamStats)

    #time.sleep(5)
    driver.quit()

    #with open("../data/league-stats/team-per-game", "w") as f:

    return

# gets recent stats for 2 teams : NEED BETTER STATS TO USE!
def getRecentStats(home,away):
    games2020 = pd.read_csv("../data/games/nba-games-2019-oct-nov-dec.txt", index_col=4)
    games2020 = games2020.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])
    print(games2020)
    return

updateData()
