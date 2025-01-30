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

# Preprocesar las variables categóricas
"""def preprocesar_datos(input_data):
    input_data['gender'] = 1 if input_data['gender'] == 'Masculino' else 0
    input_data['embarked'] = {'S': 1, 'Q': 0, 'C': 2}.get(input_data['embarked'], 1)
    input_data['pclass'] = int(input_data['pclass'])  # Asegurar que sea un entero
    input_data['cabin'] = 1 if input_data['cabin'] == 'Sí' else 0
    return input_data"""

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

    # Transformar los datos del cliente
    X_passenger= dv.transform([nuevos_datos])

    # Realizar la predicción
    y_pred_proba = model.predict_proba(X_passenger)[0][1]

# Mostrar resultado
    st.subheader("Resultado:")
    if y_pred_proba >= 0.5:
        st.success(f"El pasajero sobrevivio")
    else:
        st.error(f"El pasajero no sobrevivio ")
