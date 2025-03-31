# labraries
import pandas as pd

# this file is used to run models
# it will import models and try to run them on the data
from preprocess_data import preprocess_data_target

if __name__ == '__main__':
    data_path: str = 'data/train.csv'

    # preprocess the data
    dataset = preprocess_data_target(data_path, 'data/preprocess.csv')
    
    # import the model


    # run the model


    # try models on data/submission_test.csv (we don’t have the output of the submission_test.csv)
    pass
