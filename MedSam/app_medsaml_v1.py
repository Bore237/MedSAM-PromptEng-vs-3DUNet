import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import MedSamWrapper

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Charger  image et mask
uploaded_img = st.file_uploader("Choisir une image médicale (npy) 4 cannaux", type=["npy", "npz"])
uploaded_mask = st.file_uploader("Choisir le mask médical (npy) 4 cannaux", type=["npy", "npz"])

medsam = MedSamWrapper(
    model_type="vit_b",
    checkpoint_path="D:/marchine_learning/Projet/Segmentation/MedSam/model/medsam_vit_b.pth",
    device="cpu"
)

if uploaded_img and uploaded_mask:
    img = np.load(uploaded_img)
    mask = np.load(uploaded_mask)

    # Paramètres dans la sidebar
    slice_idx = st.sidebar.slider("Coupe", 0, img.shape[2]-1, img.shape[2]//2)
    mod_choice = st.sidebar.selectbox(
        "Sélectionner une modalité",
        ["FLAIR", "T1", "T2"],
        index=0 
    )

    # --- Utilisation selon la modalité choisie ---
    if mod_choice == "FLAIR":
        slice_img = img[:, :, slice_idx, 0]   # canal 0
        slice_mask = mask[:, :, slice_idx, 0]
    elif mod_choice == "T1":
        slice_img = img[:, :, slice_idx, 1]   # canal 1
        slice_mask = mask[:, :, slice_idx, 1]
    elif mod_choice == "T2":
        slice_img = img[:, :, slice_idx, 2]   # canal 2
        slice_mask = mask[:, :, slice_idx, 2]

    # Choix du prompt
    shape = st.sidebar.selectbox("Forme du prompt", ["Rectangle", "Point"])

    if shape == "Rectangle":
        x_min = st.sidebar.slider("x_min", 0, slice_img.shape[0]-1, 0)
        x_max = st.sidebar.slider("x_max", 0, slice_img.shape[0]-1, 60)
        y_min = st.sidebar.slider("y_min", 0, slice_img.shape[1]-1, 65)
        y_max = st.sidebar.slider("y_max", 0, slice_img.shape[1]-1, 115)
        prompt_point = None
        label_value = None
        prompt_box = np.array([(x_min, y_min), (x_max, y_max)])
        st.sidebar.write("Prompt géométrique :", prompt_box)


    elif shape == "Point":
        x_point = st.sidebar.slider("x_point", 0, slice_img.shape[0]-1, 51)
        y_point = st.sidebar.slider("y_point", 0, slice_img.shape[1]-1, 65)
        label_value = st.sidebar.selectbox("Label du point", [0, 1], index=1)
        prompt_box = None
        prompt_point = np.array([[x_point, y_point]])   # coordonnées du point
        label_value = np.array([label_value])  
        st.sidebar.write("Prompt géométrique :", prompt_point)


    # Affichage image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(slice_img, cmap="gray")
    ax1.set_title(f"Image ({mod_choice}-{slice_idx})")
    ax2.imshow(slice_mask, cmap="gray")
    ax2.set_title(f"Mask ({mod_choice}-{slice_idx})")

    if shape == "Rectangle":
        rect1 = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                            linewidth=2, edgecolor='r', facecolor='none')
        rect2 = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                            linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect1)
        ax2.add_patch(rect2)

    elif shape == "Point":
        circ1 = plt.Circle((x_point, y_point), radius=3, linewidth=2, edgecolor='b', facecolor='none')
        circ2 = plt.Circle((x_point, y_point), radius=3, linewidth=2, edgecolor='b', facecolor='none')
        ax1.add_patch(circ1)
        ax2.add_patch(circ2)
    st.pyplot(fig)

    if st.button("Lancer la prédiction"):
        # Appel du modèle
        results = medsam.predict(
            slice_img,
            slice_mask,
            input_box= prompt_box,    
            points=prompt_point, 
            labels=label_value,      
        )

        # Affichage des résultats
        fig, (ax3, ax4) = plt.subplots(1, 2)

        # Masque prédit seul
        ax3.imshow(results["masks"][0], cmap="gray", alpha=0.8)
        ax3.set_title("Masque SAM")

        # Image + masque de référence + prédiction
        ax4.imshow(slice_img, cmap="gray")
        ax4.imshow(slice_mask, cmap="Blues", alpha=0.5)
        ax4.imshow(results["masks"][0], cmap="Reds", alpha=0.5)
        ax4.set_title("Image + Masques")

        st.pyplot(fig)

        # Affichage des métriques
        st.write("Dice :", results["dice"])
        st.write("IoU :", results["iou"])
        st.write(f"Temps d'inférence : {results['time']:.3f} secondes")