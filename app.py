import pandas as pd
import streamlit as st
import pickle

st.title("IRIS EDA :sunglasses:")
placeholder = st.sidebar.empty()
st.sidebar.image("https://miro.medium.com/max/700/0*QHogxF9l4hy0Xxub.png", width = 300)
placeholder.markdown("# Iris")


st.subheader("dataset")
iris = pd.read_csv('iris.csv')

st.write(iris)

if st.checkbox("Preview Dataframe"):

    if st.button("Head"):
        st.write(iris.head(6))

    if st.button("Tail"):
        st.write(iris.tail())

from joblib import load
pkl = load('model.joblib')

sl = st.text_input("Sepal length")
sw = st.text_input("Sepal width")
pl = st.text_input("Petal length")
pw = st.text_input("Petal width")

if st.button("predict"):
    model = pickle.loads(pkl)
    output = model.predict([[sl,sw,pl,pw]])
    st.success(output[0])