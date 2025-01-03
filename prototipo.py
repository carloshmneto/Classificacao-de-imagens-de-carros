import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os
import pandas as pd

st.title("Classificador de Imagens de Carros")

base_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(base_dir, "arquivos_prototipo", "classes.csv")

class_mapping_df = pd.read_csv(csv_path)

uploaded_file = st.file_uploader("Carregue uma imagem do carro", type=["jpg", "png"])
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    model_path = os.path.join(base_dir, "arquivos_prototipo", "modelo_efficientnetb0.h5")
    model = load_model(model_path)    
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    class_name = class_mapping_df.loc[class_mapping_df['model_index'] == predicted_class, 'class_name'].values[0]
    st.write(f"Classe prevista: {class_name}")
