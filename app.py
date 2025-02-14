import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

# Carregar modelo treinado
model = load_model("modelo_catdog.h5")

# Interface do Usuário
st.title("Classificador de Imagem")
st.header("Cachorros e Gatos 🐶🐱")
uploaded_file = st.file_uploader("Envie uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem carregada", use_column_width=True)

    # Preprocessamento
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predição
    prediction = model.predict(img_array)
    prob_cachorro = float(prediction[0][0]) if prediction.shape == (1, 1) else float(prediction[0][1])
    prob_gato = 1.0 - prob_cachorro

    label = "Cachorro 🐶" if prob_cachorro >= 0.5 else "Gato 🐱"

    # Exibindo as probabilidades
    st.write(f"### Resultado da Análise: {label}")
    st.write(f"**Probabilidade de ser Cachorro**: {prob_cachorro * 100:.2f}%")
    st.write(f"**Probabilidade de ser Gato**: {prob_gato * 100:.2f}%")

    # Exibindo a acurácia do modelo
    accuracy = (prob_cachorro if prob_cachorro >= 0.5 else prob_gato) * 100
    st.write(f"**Acurácia da Predição**: {accuracy:.2f}%")

   # Exibir gráfico de probabilidades
    fig, ax = plt.subplots()
    sns.barplot(x=["Cachorro", "Gato"], y=[prob_cachorro, prob_gato], palette=["blue", "orange"], ax=ax)
    ax.set_ylabel("Probabilidade")
    ax.set_title("Confiança da Predição")
    st.pyplot(fig)