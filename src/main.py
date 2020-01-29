import tensorflow
from tensorflow import keras
import predict
import create_train_model as ctm
import view_data_log as vdl
import update_data
import rank_teams
import update_data

def main(model):
    while(True):
        a = input("Enter an option [TRAIN, PREDICT, VIEWLOG, RANK, UPDATE, END]: ")
        check = a.lower()
        if check == 'train':
            whichModel = input("Enter an option [GAMELINE, DIFF]: ").lower()
            if whichModel == 'gameline':
                print("Training gameline model...")
                try:
                    ctm.createModel()
                except ValueError:
                    print("Wrong Inputs")
                print('Loading model...')
                model = keras.models.load_model("model/test_model.h5")
            elif whichModel == 'diff':
                print("Training diff model...")
                #ctm.createDifferentialModel()
                print('Loading model...')
                model = keras.models.load_model("model/test_model.h5")
            else:
                print("wrong inputs")
                continue
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
    model = keras.models.load_model("model/test_model.h5")
    main(model)
    print("BYE")
