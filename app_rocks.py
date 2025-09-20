# app_rocks.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---------------------------
# Logo de la institución
# ---------------------------
# Ruta al logo (puede ser jpg o png)
logo_path = "logo_institucion.png"  
st.image(logo_path, width=150)  # Ajusta el ancho a tu gusto

# ---------------------------
# Configuración inicial
# ---------------------------
st.title("Clasificador de Rocas")
st.write("Sube una imagen de roca y el modelo te dirá a qué clase pertenece.")

# ---------------------------
# Parámetros del modelo
# ---------------------------
IMAGE_SIZE = 224  # cambiar al tamaño usado en tu entrenamiento
model_path = "rock_classifier_model_tl.h5"
#model_path = "D:/Maestria_AI/ProyectoFinal2/ClasificadorRocas/RocksClass.Streamlit/rock_classifier_model_tl.h5"  # ruta al modelo guardado
class_names = ['Arenisca', 'Caliza', 'Carbon', 'Granito'] # reemplaza por tus clases reales

# ---------------------------
# Cargar modelo entrenado
# ---------------------------
@st.cache_resource  # cache para no cargar el modelo cada vez
def load_my_model(path):
    model = load_model(path)
    return model

model = load_my_model(model_path)

# ---------------------------
# Función de predicción
# ---------------------------
def predict_rock(img, model, image_size=IMAGE_SIZE):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # forma (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    img_array = img_array / 255.0  # normalización
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds[0])]
    confidence = round(100 * np.max(preds[0]), 2)
    return predicted_class, confidence

# ---------------------------
# Cargar imagen desde la app
# ---------------------------
uploaded_file = st.file_uploader("Elige una imagen", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen subida", use_container_width=True)
    
    predicted_class, confidence = predict_rock(img, model)
    
    st.write(f"**Predicción:** {predicted_class}")
    st.write(f"**Confianza:** {confidence}%")
    st.success("¡Clasificación completada!")