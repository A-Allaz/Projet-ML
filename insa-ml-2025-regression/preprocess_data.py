import pandas as pd
import numpy as np
import os

class PreProcessingData:
    def __init__(self, data_path: str, submission_data_path: str, target_column: str = 'co2'):
        self.data_path = data_path
        self.submission_data_path = submission_data_path


        # Load the data
        self.__load_data__()

        # Store the target column
        self.target_column = target_column

        
        # boolean for normalisation and minmax normalisation
        self.done_normalisation = False
        self.minmax = False


        # save one hot columns
        self.one_hot_columns = []



    def __load_data__(self):
        self.data = pd.read_csv(self.data_path)
        self.sub_data = pd.read_csv(self.submission_data_path)


        # loading data into input x and output y
        self.x =self.data.drop(columns=['co2'])
        self.y = self.data['co2']

        # loading submission data into input x (we don’t have the output y)
        self.sub_x = pd.read_csv(self.submission_data_path)


    # full execution
    def full_preprocess(self):

        print("Preprocessing data...")
        # ENCODING STRING COLUMNS
        self.one_hot_encoding()

        # # its either smooth target encoding or drop text columns
        # self.drop_text_columns()
        self.smooth_target_encoding()

        self.combine_hc_nox()

        self.drop_low_correlation_columns()

        self.fill_na_values()

        # NORMALISATION
        # self.normalisation_z_score() # perform normalisation on non one hot columns
        self.normalisation_min_max()

        print("Preprocessing done")


        return self



    def save_preprocess(self, save_path: str="data/preprocess.csv"):
        # save the preprocessed data
        self.x.to_csv(save_path, index=False)
        
    def drop_low_correlation_columns(self):
        dropcols = ['id', 'co', 'hc', 'nox', 'ptcl', 'hybrid', 'fuel_type']
        dropcols = [col for col in dropcols if col in self.x.columns]
        # drop the columns
        self.x = self.x.drop(columns=dropcols)
        # drop the columns from submission data
        self.sub_x = self.sub_x.drop(columns=dropcols)

    def drop_text_columns(self):
        # select text columns
        text_columns = self.x.select_dtypes(include=['object']).columns.tolist()

        # drop the columns from the data
        self.x = self.x.drop(columns=text_columns)
        # drop the columns from submission data
        self.sub_x = self.sub_x.drop(columns=text_columns)

    def normalisation_min_max(self):
        # normalise the data
        # using min max normalisation
        # for each columns that contains float or int values
        # we will use the min max normalisation, in x and y
        numerical_columns = self.x.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # remove the one hot columns
        numerical_columns = [col for col in numerical_columns if col not in self.one_hot_columns]


        for col in numerical_columns:
            # normalise the submission data
            self.sub_x[col] = (self.sub_x[col] - self.x[col].min()) / (self.x[col].max() - self.x[col].min())
            # normalise the data
            self.x[col] = (self.x[col] - self.x[col].min()) / (self.x[col].max() - self.x[col].min())

        # store the min and max values of co2 for denormalisation
        self.max_co2 = self.y.max()
        self.min_co2 = self.y.min()
        # normalise the output y
        self.y = (self.y - self.min_co2) / (self.max_co2 - self.min_co2)

        # set the minmax boolean to true
        self.minmax = True
        self.done_normalisation = True

    def normalisation_z_score(self):
        # normalise the data
        # using z-score
        # for each columns that contains float or int values
        # we will use the z-score normalisation, in x and y
        numerical_columns = self.x.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # remove the one hot columns
        numerical_columns = [col for col in numerical_columns if col not in self.one_hot_columns]

        for col in numerical_columns:
            # normalise the data
            self.x[col] = (self.x[col] - self.x[col].mean()) / self.x[col].std()
            # normalise the submission data
            self.sub_x[col] = (self.sub_x[col] - self.x[col].mean()) / self.x[col].std()
        # normalise the output y
        self.y = (self.y - self.y.mean()) / self.y.std()

        self.done_normalisation = True
        

    def denormalise_output(
            self,
            output,
    ):  
        if self.done_normalisation:
            # denormalise the output which is a co2 prediction
            # we previously stored the min and max values of co2 in the data
            # so we can denormalise the output
            if (self.minmax):
                # denormalise using min max
                output = output * (self.max_co2 - self.min_co2) + self.min_co2
            else:
                # denormalise using z-score
                output = output * self.y.std() + self.y.mean()

        return output
    
    def denormalise_mae(
            self,
            mae,
    ):
        if self.done_normalisation:
            # denormalise the mae which is a co2 prediction
            # we previously stored the min and max values of co2 in the data
            # so we can denormalise the output
            if (self.minmax):
                # denormalise using min max
                mae = mae * (self.max_co2 - self.min_co2)
            else:
                # denormalise using z-score
                mae = mae * self.y.std()
        return mae


    def combine_hc_nox(self):
        # if hc and nox are columns in the data
        if ('hc' in self.x.columns) and ('nox' in self.x.columns):
            # fill the missing values with the sum of hc and nox
            self.x['hcnox'] = self.x['hcnox'].fillna(self.x['hc'] + self.x['nox'])
            # fill the missing values in submission data
            self.sub_x['hcnox'] = self.sub_x['hcnox'].fillna(self.sub_x['hc'] + self.sub_x['nox'])
        
    def smooth_target_encoding(
            self,
            alpha: int = 10,
    ):
        # smooth target encoding on string content columns
        string_columns = self.x.select_dtypes(include=['object']).columns.tolist()
        target_global_mean = self.y.mean()

        for col in string_columns:
            stats = self.data.groupby(col)[self.target_column].agg(['mean', 'count'])
            smooth = (stats['mean'] * stats['count'] + target_global_mean * alpha) / (stats['count'] + alpha)
            # apply the smooth target encoding to the column
            self.x[col] = self.x[col].map(smooth)
            # apply the smooth target encoding to the submission data
            self.sub_x[col] = self.sub_x[col].map(smooth)


    def fill_na_values(self):
        # fill the na values with the mean of the column
        numerical_columns = self.x.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col in numerical_columns:
            self.sub_x[col] = self.sub_x[col].fillna(self.sub_x[col].mean())
            # fill the submission data
            self.x[col] = self.x[col].fillna(self.x[col].mean())

    def one_hot_encoding(self, threshold: int = 28):
        # for each column that contains less than threshold different values
        # we will create a new column for each value and attribute 1 or 0 to the column

        # for example if we have a column with 3 different values (a, b, c)
        # we will create 3 columns (a, b, c) and attribute 1 or 0 to the column


        # Identify categorical columns with a low number of unique values
        categorical_columns = self.x.select_dtypes(include=['object']).columns.tolist()

        for col in categorical_columns:
            unique_values = self.x[col].nunique()
            if unique_values <= threshold:
                # Perform one-hot encoding
                one_hot = pd.get_dummies(self.x[col], prefix=col)
                one_hot_sub = pd.get_dummies(self.sub_x[col], prefix=col)

                # Convert boolean columns to integers (0 and 1)
                one_hot = one_hot.astype(int)
                one_hot_sub = one_hot_sub.astype(int)

                # Store the one-hot columns for later use
                self.one_hot_columns.extend(one_hot.columns.tolist())

                # Drop the original column
                self.x = self.x.drop(columns=[col])
                self.sub_x = self.sub_x.drop(columns=[col])

                # Concatenate the one-hot encoded columns
                self.x = pd.concat([self.x, one_hot], axis=1)
                self.sub_x = pd.concat([self.sub_x, one_hot_sub], axis=1)

                # Ensure both datasets have the same one-hot encoded columns
                missing_cols = set(one_hot.columns) - set(one_hot_sub.columns)
                for c in missing_cols:
                    self.sub_x[c] = 0
                self.sub_x = self.sub_x[self.x.columns]




def preprocess_data_target(data_path, submission_data_path, normalisation, save_path:str = None):
    data = pd.read_csv(data_path)
    submission_data = pd.read_csv(submission_data_path)


    # On récupère les colonnes dont les valeurs sont des chaînes de caractères
    string_columns = data.select_dtypes(include=['object']).columns
    
    # On récupère les colonnes qui ont plus d’une valeur unique
    high_cardinality_columns = [col for col in string_columns if data[col].nunique() > 1]


    # On Récupère la moyenne d’émission de CO2 pour chaque valeur unique des colonnes à haute cardinalité
    mean_co2_per_value = {}
    for col in high_cardinality_columns:
        mean_co2_per_value[col] = data.groupby(col)['co2'].mean()

    # On remplace les valeurs des colonnes à haute cardinalité par la moyenne d’émission de CO2 correspondante
    for col in high_cardinality_columns:
        data[col] = data[col].map(mean_co2_per_value[col])
        submission_data[col] = submission_data[col].map(mean_co2_per_value[col])

    # On remplace les valeurs manquantes par la moyenne de la colonne
    data = data.fillna(data.mean())
    submission_data = submission_data.fillna(data.mean())


    min_co2 = data['co2'].min()
    max_co2 = data['co2'].max()



    # On normalise les colonnes en mettant à l’échelle les valeurs entre 0 et 1
    if normalisation:
        for col in data.columns.drop(['id']):
            if col != 'co2':
                submission_data[col] = (submission_data[col] - data[col].min()) / (data[col].max() - data[col].min())
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
            
    
    # On sauvegarde le fichier
    if save_path:
        data.to_csv(save_path, index=False)
        submission_data.to_csv(save_path.replace('.csv', '_submission.csv'), index=False)

    return data, submission_data, min_co2, max_co2


if __name__ == '__main__':
    preprocess = PreProcessingData(
        data_path='data/train.csv',
        submission_data_path='data/submission_test.csv',
    )

    preprocess.full_preprocess()
    preprocess.save_preprocess()
    