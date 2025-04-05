# ______________________________LIBRARY IMPORTS
import pandas as pd
from sklearn import model_selection
#_______________________________PERSONAL IMPORTS
from preprocess_data import PreProcessingData
from models import LinearRegressionModel, MultiLayerFeedForwardModel, RandomForestRegressorModel, ExtraTreesRegressorModel


def save_submission(X_sub, Y_sub, save_path: str):
    """
    Save the submission data to a CSV file.
    
    :param X_sub: Feature matrix for submission
    :param Y_sub: Predicted values for submission
    :param save_path: Path to save the submission file
    """
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
    #__________________________________ DATA
    data_path: str = 'data/train.csv'
    submission_data_path: str = 'data/submission_test.csv'
    normalisation: bool = True
    print('Loading data')
    data = PreProcessingData(
        data_path=data_path,
        submission_data_path=submission_data_path,
        target_column='co2',
    ).full_preprocess()

    # split the data into train and test sets
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data.x, data.y, test_size=0.2, random_state=42)

    trained_model = False
    # ask the user what model to use
    model_choice = input('Choose a model: 1 for LinearRegression, 2 for MultiLayerFeedForward, 3 for RandomForestRegressor, 4 for ExtraTreesRegressor: ')
    match model_choice:
        case '1':
            model = LinearRegressionModel()
        case '2':
            storage_path: str = 'network_configs/1/mac_test.pth'
            model = MultiLayerFeedForwardModel(
                input_size=X_train.shape[1], 
                config_path='network_configs/1/1.json',
                save_path=storage_path
            )
            # train or load the model
            action_choice = input('Do you want to train the model (t) or load it (l)? ')
            match action_choice:
                case 't':
                    pass
                case 'l':
                    # load the model
                    model.load(path=storage_path)
                    trained_model = True
                case _:
                    print('Invalid choice. Exiting.')
                    exit()

        case '3':
            model = RandomForestRegressorModel(
                n_estimators=80,
                max_depth=None,
                criterion='squared_error',
                min_samples_leaf=1,
                n_jobs=-1,
                verbose=0,
                random_state=42,
            )
        case '4':
            model = ExtraTreesRegressorModel(
                n_estimators=205,
                max_depth=None,
                criterion='squared_error',
                min_samples_leaf=1,
                bootstrap=False,
                n_jobs=-1,
                verbose=0,
                random_state=42
            )
        case _:
            print('Invalid choice. Exiting.')
            exit()

    #__________________________________ FIT

    if model_choice == '2' and not trained_model:
        # train the model (without id column)
        model.fit(
            X_train=X_train.values, 
            Y_train=Y_train.values,
            X_validation=X_test.values,
            Y_validation=Y_test.values,
            epochs=1000,
            learning_rate=0.001,
            step_size=50,
            gamma=0.9,
            batch_size=128,
            patience=None,
            store_improvement=True,
        )
    else:
        # submit or test
        print("Model trained ...")
        action_choice = input('Do you want to submit the model (s) or test it (t)? ')
        match action_choice:
            case 's':
                if model_choice != '2':
                    X_train = data.x
                    Y_train = data.y
                    model.fit(
                        X=X_train, 
                        Y=Y_train,
                    )
                Y_sub = model.predict(
                    X=data.sub_x.values
                )
                # denormalize the predicted values
                Y_sub = data.denormalise_output(output=Y_sub.flatten())
                # save the submission
                match model_choice:
                    case '1':
                        save_path = 'submission_data/1_linear_regression.csv'
                    case '2':
                        save_path = 'submission_data/2_multilayer.csv'
                    case '3':
                        save_path = 'submission_data/3_randomforest.csv'
                    case '4':
                        save_path = 'submission_data/4_extratrees.csv'
                save_submission(data.sub_data, Y_sub, save_path)
            case 't':
                if model_choice != '2':
                    # train the model
                    X_train = data.x
                    Y_train = data.y
                    # train
                    model.fit(
                        X=X_train, 
                        Y=Y_train,
                    )
                mae = model.test(
                X=X_test.values, 
                Y=Y_test.values
                )
                print(f'MAE: {data.denormalise_mae(mae)}')

            case _:
                print('Invalid choice. Exiting.')
                exit()