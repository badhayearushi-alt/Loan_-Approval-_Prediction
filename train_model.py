import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
df=pd.read_csv("Loan Prediction Dataset.csv")
X=df.drop("Loan_Status",axis=1)
y=df["Loan_Status"]
X=pd.get_dummies(X,drop_first=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)
joblib.dump(model,"loan_model.pkl")
print("Saved loan_model.pkl")