import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import opendata

# user app to interact with the model
model = keras.models.load_model("predict_nba.h5")

#i.e "Milwaukee Bucks, Dallas Mavericks, ..."
home = input("Enter home team: ")
away = input("Enter away team: ")

inputs = opendata.getNeuralInputs(home,away)

predict = model.predict(inputs)
print(predict)
