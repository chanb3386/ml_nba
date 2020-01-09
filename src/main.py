import tensorflow
from tensorflow import keras
import predict
import create_train_model as ctm
import view_data_log as vdl
import update_data
import rank_teams
import update_data

model = keras.models.load_model("model/test_model.h5")

def main():
    while(True):
        a = input("Enter an option [TRAIN, PREDICT, VIEWLOG, RANK, UPDATE, END]: ")
        check = a.lower()
        if check == 'train':
            print("Training...")
            ctm.createModel()
        elif check == 'predict':
            while(True):
                try:
                    predict.predictNetwork(model)
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
            input1 = input("Enter an option: [ALL, TRAIN, TEST, PRED, END]: ").lower()
            if input1 == "all":
                csv = vdl.viewAllGames()
                print(csv)
            elif input1 == "train":
                csv = vdl.viewTrainData()
                print(csv)
            elif input1 == "test":
                csv = vdl.viewTestData()
                print(csv)
            elif input1 == "pred":
                vdl.viewPredData()
            elif input1 == "end":
                print("Ending...\n")
            else:
                print("Not Valid Input... FINISHING\n")
        elif check == 'rank':
            rank_teams.rankTeams(model)
        elif check == 'end':
            print("FINISHING...")
            break
        elif check == 'update':
            update_data.updateData()
        else:
            print("incorrect input")
    return

if __name__ == '__main__':
    main()
