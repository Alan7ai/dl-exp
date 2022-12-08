import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':

    data = pd.read_csv('./housing.csv', header=None, sep='\s+')
    data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                    'PTRATIO', 'B', 'LSTAT', 'PRICE']
    data.head()
    x = data.drop(['PRICE'], axis=1)
    y = data['PRICE']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

    lm = LinearRegression()
    lm.fit(x_train, y_train)

    y_train_pred = lm.predict(x_train)

    print("训练集的指标为：")
    print('R^2:', metrics.r2_score(y_train, y_train_pred))
    print('Adjusted R^2:', 1 - (1 - metrics.r2_score(y_train, y_train_pred)) * (len(y_train) - 1) / (
                len(y_train) - x_train.shape[1] - 1))
    print('MAE:', metrics.mean_absolute_error(y_train, y_train_pred))
    print('MSE:', metrics.mean_squared_error(y_train, y_train_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

    y_test_pred = lm.predict(x_test)

    print("测试集的指标为：")
    print('R^2:', metrics.r2_score(y_test, y_test_pred))
    print('Adjusted R^2:',
          1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1))
    print('MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_test_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

    plt.xlabel('index')
    plt.ylabel('price')
    plt.scatter(x=range(50), y=y_test_pred[:50], label='predict')
    plt.scatter(x=range(50), y=y_test[:50], label='real')

    plt.legend()
    plt.show()
