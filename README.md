# MedSAM-PromptEng-vs-3DUNet: Interactive Foundation Models (MedSAM) vs. Specialized 3D U-Net
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Research-blueviolet)
![Dataset](https://img.shields.io/badge/Dataset-BraTS-orange)
![Medical Imaging](https://img.shields.io/badge/Medical-Imaging-success)
![3D U-Net](https://img.shields.io/badge/Model-3D%20U--Net-informational)
![MedSAM](https://img.shields.io/badge/Foundation-MedSAM-darkgreen)

Ce dépôt rassemble un écosystème complet d'évaluation et d'entraînement pour la segmentation sémantique volumétrique de tumeurs cérébrales (Gliomes), basé sur les challenges internationaux **BraTS (Brain Tumor Segmentation)**.

## 🎯 Problématique & Vision du Projet

L'objectif de ce projet est de répondre à une question centrale en imagerie médicale computationnelle : **Un modèle de fondation biomédical guidé par l'homme peut-il être concurrencé par un réseau de neurones 3D convolutionnel classique, plus léger mais spécialisé de bout en bout ?**

---
## 🖼️ Aperçu Clinique & Architectural
```text
              [ Dataset IRM Multimodal BraTS ]
                              │
     ┌────────────────────────┴────────────────────────┐
     ▼                                                 ▼
[ Modèles de Fondation ]                          [ Modèle Dédié ]

SAM & MedSAM (Zero-Shot)                        - U-Net 3D (From Scratch)

Guidage par Prompts (Box/Points)                - Entraînement entièrement automatique
    │                                                 │
    └────────────────────────┬────────────────────────┘
▼
            [ Évaluation Comparative ]   Metrics: Dice, IoU, 
```
<Image src="img_brast.png" alt="Coupe 2D d'une image de cerveau (T1, T1CE, T2, Flair)" caption="Coupe 2D sous différentes modalitées IRM (T1, T1CE, T2, Flair)" />
<Image src="mask_brast.png" alt="Structures tumorales de BraTS (Edema, Tumor Core, Enhancing Tumor)" caption="Anatomie des sous-régions tumorales cibles du dataset BraTS (ED, NCR, ET)" />

## 📁 Structure du Répertoire & Phases du Projet

Le projet est découpé en trois grands modules indépendants mais interconnectés :
```text
├── Robustness_1_Prompt_MedSAM2/   # Phase 1: Robustesse de MedSAM
    ├── medsam2-project.ipynb          # inference et extraction des données statistique pour MedSAM2
    ├── Medsam2_note.txt               # Explication pour une configuration en local
│   └── analyse_stat_prompt.ipynb      # Analyse statistique des perturbations de prompts
├── MedSam/                            # Phase 2: Pipeline de données & Démo Streamlit
│   ├── Notebooks                      # Experimentation sur preprocessing(Extraction, filtrage), inference et evaluation
    ├── requirements.txt               # Dépendances spécifiques DL
│   └── app.py                         # Application Web de segmentation interactive & 3D mesh
└── U_net/                             # Phase 3: Entraînement d'un modèle Unet 3D
├── requirements.txt                   # Dépendances spécifiques DL
├── v0_segmentation_u_net.ipynb        # Prototype initial U-Net 3D (128³)
└── segmentation_u_net_v1.ipynb        # Optimisations, Data Augmentation & Logs
```

## 🛠️ Détail des Phases

### Phase 1 : Étude de Robustesse et Sensibilité (MedSAM)
Avant de déployer un modèle de fondation en clinique, il faut évaluer sa tolérance à l'erreur humaine. Cette phase simule les variations de clics et de boîtes englobantes (*bounding boxes*) d'un radiologue.
* **Méthodologie :** Injection de bruits géométriques et de déformations contrôlées sur les masques de vérité terrain pour générer des prompts dégradés.
* **Résultat :** Cartographie statistique de la chute du score Dice en fonction de l'imprécision du prompt.

### Phase 2 : Pipeline Volumétrique & Prototype de Démo (Streamlit)
Création des fondations de données et d'une interface graphique interactive permettant de manipuler et de visualiser la segmentation en temps réel.
* **Ingénierie de données :** Pipeline d'extraction à la volée depuis des archives `.tar` de volumes NIfTI. Sélection stratifiée de 50 patients (35 grosses tumeurs, 15 micro-lésions).
* **Interface Web :** Visualisation slice par slice, simulation de prompts en direct, calcul des métriques comparatives (MedSAM vs SAM standard) et reconstruction de la tumeur sous forme de maillage 3D interactif (*Marching Cubes*).

### Phase 3 : Entraînement d'un U-Net 3D Souverain
Développement et optimisation d'une architecture entièrement automatisée capable de s'affranchir complètement des prompts utilisateur.

* **Spécifications :** Volumes d'entrée de $128 \times 128 \times 128$ voxels, normalisation Z-score par volume.
* **Optimisation :** Combinaison hybride de fonctions de perte (Dice Loss + Cross-Entropy) pour contrer le déséquilibre de classes sévère (le fond vs la masse tumorale). Suivi des courbes sous TensorBoard.

---

## 📊 Matrice Comparative Théorique / Pratique

| Approche | Type de Modèle | Intrants requis à l'inférence | Automatisation | Points Forts | Points Faibles |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MedSAM / SAM** | Fondation (Zéro-shot) | Image + Bounding Box / Points | ❌ Semi-manuelle | Précision immédiate si le prompt est parfait. | Sensible au bruit, coûteux en temps par patient. |
| **U-Net 3D** | Spécialisé (Dédié) | Image IRM brute uniquement |  Automatique | Autonomie complète, ultra-rapide (~2-5s), adapté aux spécificités 3D. | Nécessite un entraînement lourd et un dataset annoté. |

---

## 🚀 Comment Démarrer et Exploiter ce Repo

### 1. Clonage et Environnement
```bash
git clone [https://github.com/VOTRE_PSEUDO/MedSAM-PromptEng-vs-3DUNet.git](https://github.com/VOTRE_PSEUDO/MedSAM-PromptEng-vs-3DUNet.git)
cd MedSAM-PromptEng-vs-3DUNet
```
### 2. Tester l'Interface Interactive (Phase 2)
Allez dans le répertoire dédié, assurez-vous d'avoir téléchargé les checkpoints de SAM/MedSAM, puis lancez l'application :

```bash
cd 2_Interactive_Pipeline_App
pip install -r ../U_net/requirements.txt streamlit plotly scikit-image nibabel
streamlit run app.py
```

### 3. Entraîner ou Améliorer le U-Net 3D (Phase 3)
Explorez les carnets Jupyter de la Phase 3 pour lancer les pipelines d'entraînement PyTorch sur GPU :
```bash
cd ../3_Custom_3D_UNet
jupyter notebook segmentation_u_net_v1.ipynb
```
---


## ✍️ Informations Projet

- **Auteur** : Goudjou Borel (Bore237)
- **Date de réalisation** : Fev 2026
- **Licence** : MIT 
