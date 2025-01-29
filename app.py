import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Cargar el modelo guardado
@st.cache
def cargar_modelo():
    with open('model.pck', 'rb') as file:
        return pickle.load(file)

# Cargar el modelo
modelo_regresion = cargar_modelo()

# Título de la app
st.title("Predicción de Titanic - Modelo de Regresión Logistica")

# Explicación de la app
st.write("""
Esta aplicación utiliza un modelo de regresión logistica entrenado sobre el dataset Titanic para predecir si el pasajero sobrevivío al accidente.
Introduce los valores de las variables para hacer una predicción.
""")

# Entradas del usuario
st.sidebar.header("Introduce las características del pasajero")

# Variables categóricas y numéricas
gender = st.sidebar.selectbox("Género", ['Femenino', 'Masculino'])
sibsp = st.sidebar.number_input("Número de hijos y/o esposo abordo", min_value=0, max_value=10)
parch = st.sidebar.number_input("Número de parientes abordo", min_value=0, max_value=10)
age = st.sidebar.number_input("Edad", min_value=0, max_value=90, value=12)
embarked = st.sidebar.selectbox("Puerto de embarque", ['S', 'Q', 'C'])
pclass = st.sidebar.selectbox("Clase social", ["1", "2", "3"])
fare = st.sidebar.number_input("Tarifa", min_value=0, value=70)
cabin = st.sidebar.selectbox("Tiene cabina", ['Sí', 'No'])

# Preprocesar las variables categóricas
def preprocesar_datos(input_data):
    input_data['gender'] = 1 if input_data['gender'] == 'Masculino' else 0
    input_data['embarked'] = 1 if input_data['embarked'] == 'S' else (0 if input_data['embarked'] == 'Q' else 2)
    input_data['pclass'] = 1 if input_data['pclass'] == '1' else (2 if input_data['pclass'] == '2' else 3)
    input_data['cabin'] = 1 if input_data['cabin'] == 'Sí' else 0
   
    
    return input_data

# Crear un DataFrame con los datos introducidos
nuevos_datos = pd.DataFrame({
    'gender': [gender],
    'age' : [age],
    'sibsp': [sibsp],
    'parch': [parch],
    'fare': [fare],
    'embarked': [embarked],
    'pclass': [pclass],
    'cabin': [cabin]
})

# Preprocesar los datos antes de hacer la predicción
nuevos_datos_procesados = preprocesar_datos(nuevos_datos)

# Realizar la predicción con el modelo cargado
if st.sidebar.button('Predecir'):
    prediccion = modelo_regresion.predict(nuevos_datos_procesados)
    
    # Mostrar el resultado
    st.write(f"La predicción del modelo es: {prediccion[0]:.2f}")
