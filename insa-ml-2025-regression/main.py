# labraries
import pandas as pd
from sklearn import model_selection
import torch
from torch import nn
import numpy as np
# this file is used to run models
# it will import models and try to run them on the data
from preprocess_data import PreProcessingData
from models import LinearRegressionModel, MultiLayerFeedForwardModel, RandomForestRegressorModel

def save_submission(X_sub, Y_sub, save_path: str):
    """
    Save the submission data to a CSV file.
    
    :param X_sub: Feature matrix for submission
    :param Y_sub: Predicted values for submission
    :param datapath: Path to save the submission file
    """
    # Y_sub has to be rounded
    Y_sub = Y_sub.round().astype(int)
    print(Y_sub)
    submission = pd.DataFrame({
        'id': X_sub['id'],
        'co2': Y_sub
    })
    # save the submission
    submission.to_csv(save_path, index=False)
    print(f'Submission saved to {save_path}')

if __name__ == '__main__':
    data_path: str = 'data/train.csv'
    submission_data_path: str = 'data/submission_test.csv'
    normalisation: bool = True

    data = PreProcessingData(
        data_path=data_path,
        submission_data_path=submission_data_path,
        target_column='co2',
    ).full_preprocess()
    

    # # split the data into train and test sets
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data.x, data.y, test_size=0.2, random_state=42)



    #__________________________________ MODEL
    # import the model
    # model = LinearRegressionModel()
    # model = MultiLayerFeedForwardModel(
    #     input_size=X_train.shape[1], 
    #     config_path='network_configs/1/1.json',
    #     save_path=storage_path
    # )
    storage_path: str = 'network_configs/rfr/store.json'
    model = RandomForestRegressorModel(
        n_estimators=31,
        max_depth=None,
        criterion='squared_error',
        min_samples_leaf=1,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    #__________________________________ LOAD

    # model.load(path=storage_path)

    #__________________________________ FIT

    
    # train the model (without id column)
    # model.fit(
    #     X_train=X_train.values, 
    #     Y_train=Y_train.values,
    #     X_val=X_test.values,
    #     Y_val=Y_test.values,
    #     epochs=1000,
    #     learning_rate=0.001,
    #     step_size=50,
    #     gamma=0.9,
    #     batch_size=128,
    #     patience=None,
    #     store_improvement=True,
    # )
    model.fit(
        X_train, 
        Y_train
    )

    #__________________________________ STORE

    # model.store(path=storage_path)

    #___________________________________ SUBMISSION TEST

    # test the model
    mae = model.test(
        X_test=X_test.values, 
        Y_test=Y_test.values
    )
    print(f'MAE: {data.denormalise_mae(mae)}')


    # predict on the submission data
    y_sub = model.predict(
        X=data.sub_x.values
    )
    
    y_sub = data.denormalise_output(output=y_sub.flatten())
    # denormalize the predicted values

    # save the submission
    save_path = 'submission_data/rfr.csv'
    save_submission(data.sub_data, y_sub, save_path)
