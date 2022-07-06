from sklearn.preprocessing import StandardScaler
import pandas as pd

class Dataset:
    def __init__(self, train_file, test_file, scaler):
        self.scaler = scaler
        self.df_train = pd.read_csv(train_file)
        self.df_test = pd.read_csv(test_file)

    def get(self, x_features, y_features):
        x_train = self.df_train.iloc[:, x_features].values
        x_train = self.scaler.fit_transform(x_train)
        y_train = self.df_train.iloc[:, y_features].values

        x_test = self.df_test.iloc[:, x_features].values
        x_test = self.scaler.transform(x_test)
        y_test = self.df_test.iloc[:, y_features].values

        return x_train, y_train, x_test, y_test
