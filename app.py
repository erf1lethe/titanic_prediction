import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Cargar el modelo guardado
@st.cache_resource
def cargar_modelo():
    try:
        with open('model.pck', 'rb') as file:
            modelo = pickle.load(file)

            # Si el modelo es una tupla, extraemos el primer elemento
            if isinstance(modelo, tuple):
                modelo = modelo[0]

            # Verificar si el modelo tiene el método predict
            if not hasattr(modelo, "predict"):
                st.error("El archivo de modelo no contiene un modelo válido.")
                return None

            return modelo

    except FileNotFoundError:
        st.error("Error: El archivo del modelo no se encuentra. Asegúrate de que 'model.pck' está en el directorio correcto.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Cargar el modelo
modelo_regresion = cargar_modelo()

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
def preprocesar_datos(input_data):
    input_data['gender'] = 1 if input_data['gender'] == 'Masculino' else 0
    input_data['embarked'] = {'S': 1, 'Q': 0, 'C': 2}.get(input_data['embarked'], 1)
    input_data['pclass'] = int(input_data['pclass'])  # Asegurar que sea un entero
    input_data['cabin'] = 1 if input_data['cabin'] == 'Sí' else 0
    return input_data

# Crear un DataFrame con los datos introducidos
nuevos_datos = pd.DataFrame([{  # Se usa lista de diccionario para evitar problemas con DataFrame vacío
    'gender': gender,
    'age': age,
    'sibsp': sibsp,
    'parch': parch,
    'fare': fare,
    'embarked': embarked,
    'pclass': pclass,
    'cabin': cabin
}])

# Preprocesar los datos
nuevos_datos_procesados = preprocesar_datos(nuevos_datos.iloc[0].to_dict())
nuevos_datos_procesados = pd.DataFrame([nuevos_datos_procesados])

# Verificar que el modelo se cargó correctamente antes de predecir
if modelo_regresion and st.sidebar.button('Predecir'):
    try:
        prediccion = modelo_regresion.predict(nuevos_datos_procesados)
        st.write(f"La predicción del modelo es: {'Sobrevivió' if prediccion[0] == 1 else 'No sobrevivió'}")
    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")
