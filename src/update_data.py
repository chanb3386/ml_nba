from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from io import StringIO


# function that updates games, team stats and opp stats files for most recent season
# WIP - games, league stats, opp stats, misc stats
def updateData():
    url = "https://www.basketball-reference.com/leagues/NBA_2020.html"
    url_games = "https://www.basketball-reference.com/leagues/NBA_2020_games.html"

    # opening the URL
    driver = webdriver.Chrome('../cdriver/chromedriver')
    driver.get(url)

    # wait
    time.sleep(2)
    actions = ActionChains(driver)

    # FINDING TEAM STATS
    teamStats = driver.find_element_by_id('all_team-stats-per_game') # need to find element better
    actions.move_to_element(teamStats)
    actions.perform()
    time.sleep(2)

    button = driver.find_element_by_xpath('/html/body/div[2]/div[5]/div[4]/div[1]/div/ul/li[2]')
    actions.move_to_element(button)
    actions.perform()
    time.sleep(2)

    csv = driver.find_element_by_xpath('/html/body/div[2]/div[5]/div[4]/div[1]/div/ul/li[2]/div/ul/li[4]/button')
    actions.move_to_element(csv)
    actions.perform()
    actions.click(csv)
    actions.perform()

    stats = driver.find_element_by_xpath('/html/body/div[2]/div[5]/div[4]/div[3]/div/div/pre')
    time.sleep(2)

    # saving CSV
    data = StringIO(stats.text)
    df = pd.read_csv(data)
    with open('../data/league-stats/team-per-game.txt', "w") as f:
        f.write(df.to_csv(index=False))

    # FINDING OPP STATS
    oppStats = driver.find_element_by_id('all_opponent-stats-per_game')
    actions.move_to_element(oppStats)
    actions.perform()
    time.sleep(2)

    button = driver.find_element_by_xpath('/html/body/div[2]/div[5]/div[5]/div[1]/div/ul/li[1]')
    actions.move_to_element(button)
    actions.perform()
    time.sleep(2)

    csv = driver.find_element_by_xpath('/html/body/div[2]/div[5]/div[5]/div[1]/div/ul/li[1]/div/ul/li[4]/button')
    actions.move_to_element(csv)
    actions.perform()
    actions.click(csv)
    actions.perform()

    stats = driver.find_element_by_xpath('/html/body/div[2]/div[5]/div[5]/div[3]/div/div/pre')
    time.sleep(2)

    # saving CSV
    data = StringIO(stats.text)
    df = pd.read_csv(data)
    with open('../data/league-stats/opponent-stats.txt', "w") as f:
        f.write(df.to_csv(index=False))

    # FINDING MISC STATS
    miscStats = driver.find_element_by_id('all_misc_stats')
    actions.move_to_element(miscStats)
    actions.perform()
    time.sleep(2)

    button = driver.find_element_by_xpath('/html/body/div[2]/div[5]/div[10]/div[1]/div/ul/li[1]')
    actions.move_to_element(button)
    actions.perform()
    time.sleep(2)

    csv = driver.find_element_by_xpath('/html/body/div[2]/div[5]/div[10]/div[1]/div/ul/li[1]/div/ul/li[4]/button')
    actions.move_to_element(csv)
    actions.perform()
    actions.click(csv)
    actions.perform()

    stats = driver.find_element_by_xpath('/html/body/div[2]/div[5]/div[10]/div[3]/div/div/pre')
    time.sleep(2)

    # saving CSV
    data = StringIO(stats.text)
    df = pd.read_csv(data)
    with open('../data/league-stats/misc-team.txt', "w") as f:
        f.write(df.to_csv(index=False))

    driver.quit()

    return

# gets recent stats for 2 teams : NEED BETTER STATS TO USE!
def getRecentStats(home,away):
    games2020 = pd.read_csv("../data/games/nba-games-2019-oct-nov-dec.txt", index_col=4)
    games2020 = games2020.drop(columns=['Start (ET)','Unnamed: 6','Unnamed: 7', 'Attend.', 'Notes'])
    print(games2020)
    return
