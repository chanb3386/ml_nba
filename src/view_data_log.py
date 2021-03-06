import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# helper functions to view data in /data_log
def viewAllGames():
    # all games tested/trained over
    nbagames = pd.read_csv("data_logs/nba_games.txt")
    return nbagames

def viewTrainData():
    # training games and labels
    traingames = pd.read_csv("data_logs/train_data.txt")
    trainlabels = pd.read_csv("data_logs/train_labels.txt",names=['labels'])
    traingames = traingames.join(trainlabels)
    return traingames

def viewTestData():
    # has test games, test labels and predictions for each game from the model
    testgames = pd.read_csv("data_logs/test_data.txt")
    testlabel = pd.read_csv("data_logs/test_labels.txt",names=['labels'])
    testpred = pd.read_csv("data_logs/test_predict.txt",names=['PRED'])
    testgames = testgames.join(testlabel)
    testgames = testgames.join(testpred)
    return testgames

def viewPredData():
    # teams and inputs
    preddata = open("data_logs/preds.txt", "r+")
    print(preddata.read())
    return

def viewGameSims():
    games = open("data_logs/games_sim.txt", "r+")
    print(games.read())
    return
