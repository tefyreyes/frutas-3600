import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Carga de tu modelo de reconocimiento de frutas
model = tf.keras.models.load_model('app.ipynb')

model.save('modelo_convertido.h5')

model = tf.keras.models.load_model('modelo_convertido.h5')


# Etiquetas de las clases de frutas (ajusta según tu modelo)
etiquetas = ['Banana', 'Berenjena','Cebolla', 'Cereza', 'Choclo', 'Cocos', 'Coliflor' , 'Frutilla', 'Kiwi', 'Limon', 'Manzana', 'Morron', ' Naranja', 'Palta', 'Papa', 'Pepino', 'Pera', 'Piña', 'Repollo', 'Sandia', 'Tomate', 'Uva', 'Zanahoria', 'Zucchini']  # Ejemplo de etiquetas

st.title('Clasificador de Frutas')

uploaded_file = st.file_uploader("Carga una imagen de una fruta", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)

    # Preprocesamiento de la imagen
    img_array = np.array(image.resize((64, 64)))  # Ajustar el tamaño según tu modelo
    img_array = img_array / 255.0  # Normalización si es necesario
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    predictions = model.predict(img_array)
    indice_prediccion = np.argmax(predictions)
    fruta_predicha = etiquetas[indice_prediccion]

    st.write("Esta fruta es probablemente:", fruta_predicha)
    st.write("Confianza:", round(predictions[0][indice_prediccion] * 100, 2), "%")
        