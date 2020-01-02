import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import opendata

# user app to interact with the model and receive predictions

model = keras.models.load_model("model/test_model.h5")

# Function to convert
def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += str(ele)
        str1 += " "

    # return string
    return str1

def predictNetwork():
    home_away = input("Enter [HOME | AWAY]: ")

    #i.e "Milwaukee Bucks, Dallas Mavericks, ..."
    if home_away == "HOME":
        home = input("Enter home team: ")
        away = input("Enter away team: ")
    elif home_away == "AWAY":
        away = input("Enter away team: ")
        home = input("Enter home team: ")
    else:
        raise NameError('Wrong inputs')

    inputs = opendata.getNeuralInputs(home,away,home_away)
    preds = open("data_logs/preds.txt","w")
    preds.write(home)
    preds.write("\n")
    preds.write(away)
    preds.write("\n")
    preds.write(listToString(inputs[0]))


    predict = model.predict(inputs)
    print("%HOME WIN: " + str(predict[0][0]))
    print("%AWAY WIN: " + str(predict[0][1]))

if __name__ == "__main__":
    while(True):
        try:
            predictNetwork()
        except KeyError:
            print("Wrong teams")
            continue
        except NameError:
            print("Wrong inputs")
            continue
        stop = input("Go again? Y/N: ")
        if stop.lower() == "n":
            print("FINISHING...\n")
            break
        elif stop.lower() == "y":
            continue
        else:
            break
