import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import predict
import create_train_model as ctm
import view_data_log as vdl
import update_data

def main():
    a = input("Enter an option [TRAIN, PREDICT, VIEWLOG, RANK, UPDATE]: ")
    check = a.lower()
    if check == 'train':
        print("Training...")
        ctm.createModel()
    elif check == 'predict':
        while(True):
            try:
                predict.predictNetwork()
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
    elif check == 'viewlog':
        input1 = input("Enter an option: [ALL, TRAIN, TEST, PRED, END]: ")
        if input1 == "ALL":
            csv = vdl.viewAllGames()
            print(csv)
        elif input1 == "TRAIN":
            csv = vdl.viewTrainData()
            print(csv)
        elif input1 == "TEST":
            csv = vdl.viewTestData()
            print(csv)
        elif input1 == "PRED":
            vdl.viewPredData()
        elif input1 == "END":
            print("Ending...\n")
        else:
            print("Not Valid Input... FINISHING\n")
    elif check == 'rank':
        rank_teams.rankTeams()
    else:
        print("Incorrect Input")
    return

if __name__ == '__main__':
    while(True):
        input2 = input("Run? y/n: ")
        if input2 == 'y':
            main()
        else:
            break
