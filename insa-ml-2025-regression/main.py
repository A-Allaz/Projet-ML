# labraries
import pandas as pd
from sklearn import model_selection
# this file is used to run models
# it will import models and try to run them on the data
from preprocess_data import preprocess_data_target
from models import LinearRegressionModel, MultiLayerFeedForwardModel

def get_datasets(data_path: str, submission_data_path: str, column_target: str, normalisation: bool = True):
    """
    Arrange the dataset by dropping the target column and returning the features and target separately.
    
    :param dataset: DataFrame containing the dataset
    :param column_target: The name of the target column to be dropped
    :return: Tuple of features (X) and target (Y)
    """
    dataset, X_submission, min_co2, max_co2 = preprocess_data_target(
        data_path=data_path,
        submission_data_path=submission_data_path,
        normalisation=normalisation,
        save_path='data/1.csv')

    X = dataset.drop(columns=[column_target])
    Y = dataset[column_target]


    return X, Y, X_submission, min_co2, max_co2

def save_submission(X_sub, Y_sub, save_path: str):
    """
    Save the submission data to a CSV file.
    
    :param X_sub: Feature matrix for submission
    :param Y_sub: Predicted values for submission
    :param datapath: Path to save the submission file
    """
    # return format
    # id,co2
    # 41257,527
    # 41258,196
    # 41259,457
    # etc.

    # Y_sub has to be rounded
    Y_sub = Y_sub.round().astype(int)
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
    normalisation: bool = False

    # preprocess the data
    X, Y, X_submission, min_co2, max_co2 = get_datasets(
        data_path=data_path,
        submission_data_path=submission_data_path,
        column_target='co2',
        normalisation=normalisation
    )

    # split the data into train and test sets
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.0001, random_state=42)


    # # import the model
    # model = LinearRegressionModel()
    model = MultiLayerFeedForwardModel(input_size=X_train.drop(columns=['id']).shape[1], config_path='network_configs/1.json')
    
    # train the model (without id column)
    model.fit(X_train.drop(columns=['id']), Y_train, epochs=10000)

    # # test the model
    # mae = model.test(X_test.drop(columns=['id']), Y_test)
    # print(f'Mean Absolute Error: {mae}')

    # # export for submission

    # # predict on the submission data
    # Y_submission_norm = model.predict(X_submission.drop(columns=['id']))
    # # denormalize the predicted values
    # Y_submission = Y_submission_norm
    # if normalisation:
    #     Y_submission = Y_submission_norm * (max_co2 - min_co2) + min_co2


    # # save the submission
    # save_path = 'submission_data/linear_regression_unormalised.csv'
    # save_submission(X_submission, Y_submission, save_path)

    # save the model
