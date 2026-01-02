import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@st.cache_resource
def get_model():
    titanic = sns.load_dataset('titanic')
    titanic = titanic.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'])
    titanic['age'].fillna(titanic['age'].median(), inplace=True)
    titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
    titanic = pd.get_dummies(titanic, columns=['sex', 'embarked'], drop_first=True)
    titanic.dropna(subset=['fare'], inplace=True)
    
    X = titanic.drop(columns=['survived'])
    y = titanic['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model

if __name__ == "__main__":
    print("Titanic AI Model")