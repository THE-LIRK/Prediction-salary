import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt



st.write("Prédiction de salaire")
st.sidebar.header('Rentrez vos informartions')


df1=pd.read_csv("ds_salaries.csv")
df1.drop('Unnamed: 0',axis=1,inplace=True)
df1.drop('salary_in_usd',axis=1,inplace=True)
#replace
df1["experience_level"]=df1["experience_level"].str.replace("EN", "Entry-Level")
df1["experience_level"]=df1["experience_level"].str.replace("MI", "Junior Mid-Level")
df1["experience_level"]=df1["experience_level"].str.replace("SE", "Intermediate Senior-Level")
df1["experience_level"]=df1["experience_level"].str.replace("EX", "Expert Executive-Level / Director")



df1["employment_type"]=df1["employment_type"].str.replace("PT", "Part-time")
df1["employment_type"]=df1["employment_type"].str.replace("FT", "Full-time")
df1["employment_type"]=df1["employment_type"].str.replace("CT", "Contract")
df1["employment_type"]=df1["employment_type"].str.replace("FL", "Freelance")

df2=df1



# Loads the salary Dataset

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df1,test_size=0.2,random_state=38)

train= df1.drop('salary',axis=1)
train_labels = df1['salary'].copy()

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline



# Sidebar
# Header of Specify Input Parameters


data={}
def client_caract_entree():
            
            for x in df1.drop('salary',axis=1).columns:
                k=st.sidebar.selectbox(x,(df1.drop('salary',axis=1)[x].unique()))
                data[df1[x].name]=k
            profil_client=pd.DataFrame(data,index=[0])
            return profil_client  

df=client_caract_entree()
dt=pd.concat([df,train],axis=0)


# Main Panel

# Print specified input parameters
st.header('Vos informations')
st.write(df)
st.write('---')

pipeline = Pipeline([
    ('one_hot_cat',OneHotEncoder())
])

train_pipelined=pipeline.fit_transform(train)
dt=pipeline.fit_transform(dt)
df=dt[:1]



from sklearn.tree import DecisionTreeRegressor
# Build Regression Model
model = DecisionTreeRegressor(random_state=0)
model.fit(train_pipelined,train_labels)

# Apply Model to Make Prediction

prediction = model.predict(df)


st.header('Prediction')
st.write(prediction)
st.write('---')




   


# fig = px.histogram(df2, x='experience_level',y='salary',color='experience_level',histfunc="avg",text_auto=True, title='Moyenne d').update_xaxes(categoryorder='total descending')
# st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# fig = px.histogram(df2, x='experience_level',y='salary',color='company_size',histfunc="avg",text_auto=True, title='Moyenne d').update_xaxes(categoryorder='total descending')
# st.plotly_chart(fig, theme="streamlit", use_container_width=True)