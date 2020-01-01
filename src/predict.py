import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import opendata

# user app to interact with the model
model = keras.models.load_model("model/test_model.h5")

def predictNetwork():
    home_away = input("Enter [HOME | AWAY]: ")



    #i.e "Milwaukee Bucks, Dallas Mavericks, ..."
    if home_away == "HOME":
        home = input("Enter home team: ")
        away = input("Enter away team: ")
    elif home_away == "AWAY":
        away = input("Enter away team: ")
        home = input("Enter home team: ")

    inputs = opendata.getNeuralInputs(home,away,home_away)
    #np.savetxt("prediction_results/pred_inputs.txt", inputs)


    predict = model.predict(inputs)
    print("%HOME WIN: " + str(predict[0][0]))
    print("%AWAY WIN: " + str(predict[0][1]))

if __name__ == "__main__":
    while(True):
        predictNetwork()
        stop = input("Go again? Y/N: ")
        if stop == "N":
            print("FINISHING...\n")
            break
        else:
            continue
