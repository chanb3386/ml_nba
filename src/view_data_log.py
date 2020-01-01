import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# helper functions to view data in /data_log
def viewAllGames():
    nbagames = pd.read_csv("data_logs/nba_games.txt")
    return nbagames

def viewTrainData():
    traingames = pd.read_csv("data_logs/train_data.txt")
    trainlabels = pd.read_csv("data_logs/train_labels.txt")
    traingames = traingames.join(trainlabels)
    return traingames

def viewTestData():
    testgames = pd.read_csv("data_logs/test_data.txt")
    testlabel = pd.read_csv("data_logs/test_labels.txt",names=['labels'])
    testpred = pd.read_csv("data_logs/test_predict.txt",names=['PRED'])
    testgames = testgames.join(testlabel)
    testgames = testgames.join(testpred)
    return testgames

def viewPredData():
    preddata = open("data_logs/preds.txt", "r+")
    print(preddata.read())
    return

if __name__ == "__main__":
    input = input("Enter an option: [ALL, TRAIN, TEST, PRED, END]: ")
    if input == "ALL":
        csv = viewAllGames()
        print(csv)
    elif input == "TRAIN":
        csv = viewTrainData()
        print(csv)
    elif input == "TEST":
        csv = viewTestData()
        print(csv)
    elif input == "PRED":
        viewPredData()
    elif input == "END":
        print("Ending...\n")
    else:
        print("Not Valid Input... FINISHING\n")
