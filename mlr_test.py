from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn import model_selection
import constants
import matplotlib.pyplot as plt
import spec_utils


def get_X_y_from_df(df):
    columns = list(df.columns)
    cols_to_remove = ["id","oc","lc1","lu1"]
    cols_to_remove = ["id","oc","lc1","lu1","phh","phc","ec","caco3","p","n","k","elevation","stones"]
    for c in cols_to_remove:
        columns.remove(c)
    X = df[columns].to_numpy()
    y = df[["oc"]].to_numpy()
    return X,y.reshape(-1)


df = pd.read_csv(constants.DATASET)
train_df, test_df = model_selection.train_test_split(df, test_size=0.2)
train_X = train_df[spec_utils.get_wavelengths()].to_numpy()
train_y = train_df[["oc"]].to_numpy()
test_X = test_df[spec_utils.get_wavelengths()].to_numpy()
test_y = test_df[["oc"]].to_numpy()

model = LinearRegression()
model.fit(train_X, train_y)
print(model.score(train_X, train_y))
print(model.score(test_X, test_y))

X = train_X[0]
plt.plot(X)
plt.show()
plt.scatter(list(range(len(X))),X)
plt.show()




