import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# Cargar el modelo y el DictVectorizer
with open('model.pck', 'rb') as f:
    dv, model = pickle.load(f)

# Verificar que el modelo y el vectorizador son correctos
st.write(f"Tipo de DictVectorizer: {type(dv)}")
st.write(f"Tipo de modelo: {type(model)}")
st.write("Características que espera el DictVectorizer:")
st.write(dv.get_feature_names_out())

st.title("Predicción de Titanic - Modelo de Regresión Logística")

st.write("""
Esta aplicación utiliza un modelo de regresión logística entrenado sobre el dataset Titanic para predecir si el pasajero sobrevivió al accidente.
Introduce los valores de las variables para hacer una predicción.
""")

# Entradas del usuario
st.sidebar.header("Introduce las características del pasajero")

Gender = st.sidebar.selectbox("Género", ['Femenino', 'Masculino'])
SibSp = st.sidebar.number_input("Número de hijos y/o esposo a bordo", min_value=0, max_value=10,value=0)
Parch = st.sidebar.number_input("Número de parientes a bordo", min_value=0, max_value=10, value=0)
Age = st.sidebar.number_input("Edad", min_value=0, max_value=90, step=1, value=30)
Embarked = st.sidebar.selectbox("Puerto de embarque", ['S', 'Q', 'C'])
Pclass = st.sidebar.selectbox("Clase social", [1, 2, 3])
Fare = st.sidebar.number_input("Tarifa", min_value=0.000, max_value=1000.000, value=30.000)
Cabin = st.sidebar.selectbox("Tiene cabina", ['Sí', 'No'])

if st.button("Predecir"):
    # Convertir valores categóricos a representaciones numéricas
    if Gender == 'Masculino': 
        Gender = 1
    else: 
        Gender = 0
    if Cabin == 'Sí':
        Cabin = 1
    else: 
        Cabin=0
    Embarked = 1 if Embarked == 'Q' else (0 if Embarked == 'S' else 2)
    
    nuevos_datos = {  
        'Gender': Gender,
        'Age': Age,
        'Sibsp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked,
        'Pclass': Pclass,
        'Cabin': Cabin
    }

    # Verifica los datos antes de la transformación
    st.write("Datos del pasajero:", nuevos_datos)

    # Transformar las características del pasajero con el DictVectorizer
    X_passenger = dv.transform([nuevos_datos])

    # Realizar la predicción
    
    y_pred = model.predict_proba(X_passenger)

    # Mostrar el resultado
    st.subheader("Resultado:")
    st.success(y_pred)
   



