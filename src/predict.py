import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import opendata

# user app to interact with the model and receive predictions

# takes user input to predict a game
def predictNetwork(model):
    home_away = input("Enter [HOME | AWAY]: ")

    if home_away.lower() == "home":
        # i.e "Milwaukee Bucks, Dallas Mavericks, ..."
        home = input("Enter home team: ")
        away = input("Enter away team: ")
    elif home_away.lower() == "away":
        away = input("Enter away team: ")
        home = input("Enter home team: ")
    else:
        raise NameError('Wrong inputs')

    try:
        type = input("Enter a model [ML, DIFF]: ").lower()
    except:
        print("Wrong Input")

    inputs = opendata.getNeuralInputs(home,away,home_away)
    preds = open("data_logs/preds.txt","w")
    preds.write(home)
    preds.write("\n")
    preds.write(away)
    preds.write("\n")
    preds.write(opendata.listToString(inputs[0]))


    predict = model.predict(inputs)

    if type == 'ml':
        if home_away.lower() == "home":
            print("%HOME WIN: " + str(predict[0][0]))
            print("%AWAY WIN: " + str(predict[0][1]))
        else:
            print("%AWAY WIN: " + str(predict[0][0]))
            print("%HOME WIN: " + str(predict[0][1]))
    elif type == 'diff':
        print(predict)
        print(str(predict[0]))
    else:
        raise NameError('Wrong Inputs')

# used in rank_teams.py
def predictNetwork2(home,away,model):
    inputs = opendata.getNeuralInputs(home,away,"HOME")
    predict = model.predict(inputs)
    return predict
