import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

#dataset
penguins=pd.read_csv('data/penguins_cleaned.csv')
#copy the dataset as dataframe (df)
df=penguins.copy()
df
#fixed target variable
target = 'species'
#encodeer variable
encode=['sex', 'island']
encode

for col in encode:
    dummy=pd.get_dummies(df[col], prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]
    
df
target_mapper={'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(value):
    return target_mapper[value]

df['species']=df['species'].apply(target_encode)
df

#Separating X and y
X=df.drop('species', axis=1)
X
y=df['species']
y

#Build random forest model

clf=RandomForestClassifier()
clf.fit(X,y)

# saving the model
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))

