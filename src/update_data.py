from selenium import webdriver
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


# function that updates team stats and opp stats files for most recent season
# WIP
def updateData():
    url = "https://www.basketball-reference.com/leagues/NBA_2020.html"
    driver = webdriver.Chrome('../cdriver/chromedriver')
    driver.get(url)
    time.sleep(5)
    driver.quit()
    return

# gets recent stats for 2 teams
def getRecentStats(home,away):
    games2020 = pd.read_csv("../data/games/nba-games-2019-oct-nov-dec.txt", index_col=4)
    games2020 = games2020.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])
    print(games2020)
    return

getRecentStats(1,2)
