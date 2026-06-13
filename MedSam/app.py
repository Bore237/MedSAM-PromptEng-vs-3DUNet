import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import MedSamWrapper
from skimage import measure
import plotly.graph_objects as go 
from skimage.morphology import closing, opening, disk
from skimage.morphology import remove_small_objects

def intensity_range(slice_ref, mask_ref, p_min=10, p_max=90):
    tumor_pixels = slice_ref[mask_ref > 0]
    return (
        np.percentile(tumor_pixels, p_min),
        np.percentile(tumor_pixels, p_max)
    )

def propagate(slice_curr, ref_inst, size_close=7):
    appearance = (slice_curr >= ref_inst[0]) & (slice_curr <= ref_inst[1])
    search_area = closing(appearance, footprint=disk(size_close))
    #mask_clean = opening(search_area, footprint=disk(3))
    mask_clean = remove_small_objects(search_area, min_size=30)
    return mask_clean

# Charger  image et mask
uploaded_img = st.file_uploader("Choisir une image médicale (npy) 4 cannaux", type=["npy", "npz"])
col1, col2 = st.columns(2)
#uploaded_mask = st.file_uploader("Choisir le mask médical (npy) 4 cannaux", type=["npy", "npz"])

medsam = MedSamWrapper(
    model_type="vit_b",
    checkpoint_path="D:/marchine_learning/Projet/Segmentation/MedSam/model/medsam_vit_b.pth",
    device="cpu"
)

sam = MedSamWrapper(
    model_type="vit_b",
    checkpoint_path="D:/marchine_learning/Projet/Segmentation/MedSam/model/sam_vit_b_01ec64.pth",
    device="cpu"
)

prediction_state = [False, False]

if uploaded_img:
    imgs = np.load(uploaded_img)
    mask = imgs['seg']

    # Paramètres dans la sidebar
    slice_idx = st.sidebar.slider("Coupe", 0, mask.shape[2]-1, mask.shape[2]//2)
    mod_choice = st.sidebar.selectbox(
        "Sélectionner une modalité",
        ["flair", "t1ce", "t2"],
        index=0 
    )

    type_tumeur = st.sidebar.selectbox(
        "Sélectionner une modalité",
        ["0", "1", "2", "4"],
        index=0 
    )

    # --- Utilisation selon la modalité choisie ---
    slice_img = imgs[mod_choice][:, :, slice_idx]
    slice_mask = mask[:, :, slice_idx]
    if type_tumeur != "0":
        slice_mask = mask[:, :, slice_idx] == int(type_tumeur)
    

    # Choix du prompt
    shape = st.sidebar.selectbox("Forme du prompt", ["Rectangle", "Point"])

    if shape == "Rectangle":
        x_min = st.sidebar.slider("x_min", 0, slice_img.shape[0]-1, 35)
        x_max = st.sidebar.slider("x_max", 0, slice_img.shape[0]-1, 180)
        y_min = st.sidebar.slider("y_min", 0, slice_img.shape[1]-1, 58)
        y_max = st.sidebar.slider("y_max", 0, slice_img.shape[1]-1, 130)
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

    if st.button("Lancer prediction") or prediction_state[0]:
        prediction_state[0] = True
        # Appel du modèle
        results_medsam = medsam.predict(
            slice_img,
            slice_mask,
            input_box= prompt_box,    
            points=prompt_point, 
            labels=label_value,      
        )

        results_sam = sam.predict(
            slice_img,
            slice_mask,
            input_box= prompt_box,    
            points=prompt_point, 
            labels=label_value,      
        )

        # Affichage des résultats
        fig, (ax3, ax4) = plt.subplots(1, 2)
        # Masque prédit seul
        ax3.imshow(results_medsam["masks"][0], cmap="gray", alpha=0.8)
        ax3.set_title("Masque MedSAM")

        # Image + masque de référence + prédiction
        ax4.imshow(slice_img, cmap="gray")
        ax4.imshow(slice_mask, cmap="Blues", alpha=0.5)
        ax4.imshow(results_medsam["masks"][0], cmap="Reds", alpha=0.5)
        ax4.set_title("Image + Masques")
        st.pyplot(fig)

        # Affichage des résultats
        fig, (ax5, ax6) = plt.subplots(1, 2)
        # Masque prédit seul
        ax5.imshow(results_sam["masks"][0], cmap="gray", alpha=0.8)
        ax5.set_title("Masque SAM")

        # Image + masque de référence + prédiction
        ax6.imshow(slice_img, cmap="gray")
        ax6.imshow(slice_mask, cmap="Blues", alpha=0.5)
        ax6.imshow(results_sam["masks"][0], cmap="Reds", alpha=0.5)
        ax6.set_title("Image + Masques")
        st.pyplot(fig)

        # Affichage des métriques SAM et MedSAM
        st.write("Dice MedSam:", results_medsam["dice"])
        st.write("IoU MedSam:", results_medsam["iou"])
        st.write(f"Temps d'inférence MedSam: {results_medsam['time']:.3f} secondes")
        st.write("Dice SAM:", results_sam["dice"])
        st.write("IoU SAM:", results_sam["iou"])
        st.write(f"Temps d'inférence SAM: {results_sam['time']:.3f} secondes")

    if st.button("Lancer la prédiction 3D") or prediction_state[1]:
        prediction_state[1] = True
        ## Redue 3D
        mask_small = mask[::2, ::2, ::2]
        #mask_small = mask
        fig1 = go.Figure()

        for val, color in [(1, 'gray'), (2, 'orange'), (4, 'red')]:
            # masque binaire pour la classe
            mask_val = (mask_small == val).astype(np.uint8)
            # si la classe n'existe pas dans le volume → skip
            if mask_val.sum() == 0:
                continue

            # extraction de la surface
            verts, faces, _, _ = measure.marching_cubes(mask_val, level=0.5)

            x, y, z = verts.T
            i, j, k = faces.T

            # ajout d'une couche Mesh3D
            fig1.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    color=color,
                    opacity=0.5,
                    name=f"Classe {val}"
                )
            )
        fig1.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))

        threshold = intensity_range(slice_img, slice_mask)
        imgd = np.zeros((240, 240, 155))
        for k in range(0, 155, 2):
            imgd_temps = imgs[mod_choice][:, :, k]
            imgd[:, :, k] = propagate(imgd_temps, threshold, 10)

        # extraction de la surface
        verts1, faces1, _, _ = measure.marching_cubes(mask_val, level=0.5)
        x1, y1, z1 = verts1.T
        i1, j1, k1 = faces1.T

        fig2 = go.Figure(
            data=[
                go.Mesh3d(
                    x=x1, y=y1, z=z1,
                    i=i1, j=j1, k=k1,
                    color='blue',
                    opacity=0.5
                )
            ]
        )
        fig2.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))

        with col1: 
            st.plotly_chart(fig1, use_container_width=True)
        with col2: 
            st.plotly_chart(fig2, use_container_width=True) 