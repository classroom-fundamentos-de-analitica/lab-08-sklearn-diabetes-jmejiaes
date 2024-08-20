import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def read_data():
    train_data, test_data = pd.read_csv('train_dataset.csv'), pd.read_csv('test_dataset.csv')
    x_train, y_train = train_data.drop('target', axis=1), train_data['target']
    x_test, y_test = test_data.drop('target', axis=1), test_data['target']

    return x_train, x_test, y_train, y_test


def train_model(data):
    x_train, x_test, y_train, y_test = data
    
    model = LinearRegression()
    
    model.fit(x_train, y_train)

    # evaluate the model in r2 for train and test
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"R2 score for train: {r2_train}")
    print(f"R2 score for test: {r2_test}")

    return model

def save_model(model):
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

def main():
    data = read_data()
    model = train_model(data)
    save_model(model)

if __name__ == '__main__':
    main()
