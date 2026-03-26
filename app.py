import streamlit as st
import os
from pathlib import Path
from PIL import Image
import numpy as np

# --- Configurations ---
st.set_page_config(page_title="Classification Cats vs Dogs", layout="centered", page_icon="🐾")

st.title("🐾 Classification d'Images : Chat vs Chien")
st.markdown("Cette application télécharge automatiquement le célèbre dataset **Cats vs Dogs**, entraîne un modèle d'Intelligence Artificielle (Réseau de Neurones Convolutif) et vous permet de **tester vos propres photos**.")

DATA_ROOT = Path("data_images")
MODELS_DIR = Path("models")
CLASSES = ["Cat", "Dog"]
IMAGE_SIZE = (128, 128)

@st.cache_resource(show_spinner="Téléchargement du dataset (version ultra-rapide) et entraînement du modèle en cours...")
def prepare_and_train_cnn():
    import tensorflow as tf
    from tensorflow import keras
    import random
    import os
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "cnn_model.keras"
    
    # Si le modèle est déjà entraîné, on le charge pour aller très vite
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
        
    random.seed(42)
    tf.random.set_seed(42)
    
    # --- 1. Téléchargement via Keras (Mini Version Fiable 68 Mo) ---
    st.toast("Téléchargement du mini-dataset Cats vs Dogs depuis la source certifiée...")
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    zip_dir = keras.utils.get_file('cats_and_dogs.zip', origin=url, extract=True)
    base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    
    BATCH_SIZE = 32
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    val_ds = keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    # Normalisation pixels (0 à 1)
    normalization_layer = keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    
    # --- 2. Construction du Modèle CNN (Deep Learning) ---
    st.toast("Construction du modèle CNN et entraînement...")
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2, activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Entraînement sur une partie du jeu de données mini
    subset_train = train_ds.take(40) # 40 * 32 = 1280 images
    subset_val = val_ds.take(10)      # 10 * 32 = 320 images
    
    model.fit(subset_train, validation_data=subset_val, epochs=5)
    
    # Sauvegarde du modèle pour ne plus avoir à réentraîner à chaque fois !
    model.save(model_path)
    return model

# ---------------------------------------------
# DÉMARRAGE ET INTERFACE
# ---------------------------------------------

try:
    cnn_model = prepare_and_train_cnn()
    st.success("✅ Modèle d'IA correctement chargé et prêt à l'emploi !")
except Exception as e:
    st.error(f"Erreur lors de la préparation : {e}")
    st.stop()

def preprocess_for_cnn(image):
    # Convertir l'image utilisateur en RGB et la mettre à la bonne dimension (128x128)
    img = image.convert("RGB").resize(IMAGE_SIZE)
    # Normalisation pixels
    arr = np.array(img).astype("float32") / 255.0
    # Ajouter la dimension pour le batch => (1, 128, 128, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr

st.markdown("---")
st.header("📸 Testez avec vos photos de Chat ou de Chien")

# Widget d'Upload d'image
uploaded_file = st.file_uploader("Faites glisser une image ici (JPG, PNG)", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Lecture de l'image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="L'image sélectionnée", use_container_width=True)

    with col2:
        st.write("Cliquez ci-dessous pour lancer l'analyse de cette photo avec le CNN que nous venons d'entraîner :")
        if st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True):
            with st.spinner("Analyse des pixels en cours..."):
                x_input = preprocess_for_cnn(image)
                probs = cnn_model.predict(x_input, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(np.max(probs)) * 100
                
                # Stylisation de l'affichage du diagnostic
                st.markdown("### Résultat :")
                if CLASSES[pred_idx] == "Cat":
                    st.success(f"🐱 **C'est un Chat (CAT) !**")
                else:
                    st.success(f"🐶 **C'est un Chien (DOG) !**")
                    
                st.info(f"📊 **Niveau de certitude de l'IA : {confidence:.1f}%**")
