import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import opendata

# user app to interact with the model
model = keras.models.load_model("model/test_model.h5")

#i.e "Milwaukee Bucks, Dallas Mavericks, ..."
home = input("Enter home team: ")
away = input("Enter away team: ")

inputs = opendata.getNeuralInputs(home,away)

predict = model.predict(inputs)
print("%HOME WIN: " + str(predict[0][0]))
print("%AWAY WIN: " + str(predict[0][1]))
