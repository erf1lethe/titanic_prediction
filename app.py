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
SibSp = st.sidebar.number_input("Número de hijos y/o esposo a bordo", min_value=0, max_value=10, step=1, value=0)
Parch = st.sidebar.number_input("Número de parientes a bordo", min_value=0, max_value=10, step=1, value=0)
Sge = st.sidebar.number_input("Edad", min_value=0, max_value=90, step=1, value=30)
Embarked = st.sidebar.selectbox("Puerto de embarque", ['S', 'Q', 'C'])
Pclass = st.sidebar.selectbox("Clase social", [1, 2, 3])
Fare = st.sidebar.number_input("Tarifa", min_value=0.0, value=30.0, step=0.1)
Cabin = st.sidebar.selectbox("Tiene cabina", ['Sí', 'No'])

if st.button("Predecir"):
    # Convertir valores categóricos a representaciones numéricas
    Gender = 1 if Gender == 'Masculino' else 0
    Cabin = 1 if Cabin == 'Sí' else 0
    Embarked_dict = {'S': 0, 'Q': 1, 'C': 2}  # Representación numérica del puerto
    Embarked = Embarked_dict.get(Embarked, 0)
    
    nuevos_datos = {  
        'Gender': Gender,
        'Age': Age,
        'Sibsp': SibSp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked,
        'Pclass': pclass,
        'Cabin': cabin
    }

    # Verifica los datos antes de la transformación
    st.write("Datos del pasajero:", nuevos_datos)

    # Transformar las características del pasajero con el DictVectorizer
    X_passenger = dv.transform([nuevos_datos])

    # Realizar la predicción
    y_pred = model.predict_proba(X_passenger)[:, 1]

    # Mostrar el resultado
    st.subheader("Resultado:")
    if y_pred >= 0.5:
        st.success(f"El pasajero sobrevivió con una probabilidad de {y_pred[0]*100:.2f}%")
    else:
        st.error(f"El pasajero no sobrevivió con una probabilidad de {(1-y_pred[0])*100:.2f}%")



