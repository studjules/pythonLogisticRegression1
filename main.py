import pandas as pd

#import heart dataset

df = pd.read_csv('C:\\Users\\marce\\Downloads\\Heart.csv')

# print df.head())
print(df.head())
# print df.info())
print(df.info())

df = df.drop(['Unnamed: 0'], axis=1)
print(df)

df["ChestPain"] = df["ChestPain"].astype('category')
df["ChestPain"] = df["ChestPain"].cat.codes


df["Thal"] = df["Thal"].astype('category')
df["Thal"] = df["Thal"].cat.codes


df["AHD"] = df["AHD"].astype('category')
df["AHD"] = df["AHD"].cat.codes
print(df)

print(df.isnull().sum())

df = df.dropna()
print(df)

X=df.drop(columns = ['AHD'])
y=df['AHD']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21)
print(X_train.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#Scale Data for better performance
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##import Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = logreg.predict(X_test_scaled)

print(y_pred)

#Performance Metrics
print(logreg.score(X_train_scaled, y_train))
print(logreg.score(X_test_scaled, y_test))

#add parameters
logreg1= LogisticRegression(random_state=0,
C=1, fit_intercept=True).fit(X_train_scaled, y_train)

print(logreg1.score(X_train_scaled, y_train))
print(logreg1.score(X_test_scaled, y_test))