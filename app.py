import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Cargar el modelo y el DictVectorizer
with open('model.pck', 'rb') as f:
    dv, model = pickle.load(f)

st.title("Predicción de Titanic - Modelo de Regresión Logística")

st.write("""
Esta aplicación utiliza un modelo de regresión logística entrenado sobre el dataset Titanic para predecir si el pasajero sobrevivió al accidente.
Introduce los valores de las variables para hacer una predicción.
""")

# Entradas del usuario
st.sidebar.header("Introduce las características del pasajero")

gender = st.sidebar.selectbox("Género", ['Femenino', 'Masculino'])
sibsp = st.sidebar.number_input("Número de hijos y/o esposo a bordo", min_value=0, max_value=10, step=1, value=0)
parch = st.sidebar.number_input("Número de parientes a bordo", min_value=0, max_value=10, step=1, value=0)
age = st.sidebar.number_input("Edad", min_value=0, max_value=90, step=1, value=30)
embarked = st.sidebar.selectbox("Puerto de embarque", ['S', 'Q', 'C'])
pclass = st.sidebar.selectbox("Clase social", [1, 2, 3])
fare = st.sidebar.number_input("Tarifa", min_value=0.0, value=30.0, step=0.1)
cabin = st.sidebar.selectbox("Tiene cabina", ['Sí', 'No'])

if st.button("Predecir"):
    nuevos_datos = {  
    'gender': gender,
    'age': age,
    'sibsp': sibsp,
    'parch': parch,
    'fare': fare,
    'embarked': embarked,
    'pclass': pclass,
    'cabin': cabin
}

    X_passenger = dv.transform([nuevos_datos])

    # Realizar la predicción
    y_pred = model.predict(X_passenger)[:,1]

# Mostrar resultado
    st.subheader("Resultado:")
    if y_pred >= 0.5:
        st.success(f"El pasajero sobrevivio")
    else:
        st.error(f"El pasajero no sobrevivio ")
