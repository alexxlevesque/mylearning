import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

bottle = pd.read_csv("kaggle/datasets/calcofi/bottle.csv") # outputs a df, no need to run pd.DataFrame()

bottle = bottle.dropna(subset=["R_POTEMP", "R_SALINITY"]) # the subset argument is used to specify which columns to drop null values from

X = bottle[["R_POTEMP"]] # features must be a 2D array
y = bottle["R_SALINITY"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color="red")
plt.show() # needed to close the plot to continue

print(r2_score(y_test, y_pred)) # 0.26 score, therefore the model is too simple and underfits the data