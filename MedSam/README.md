# Pipeline d'Imagerie 3D & Prototype de Segmentation Interactive (MedSAM)

## 📋 Description

Ce dépôt héberge un framework complet conçu pour extraire, traiter et segmenter de manière interactive des volumes IRM tridimensionnels de tumeurs cérébrales (Gliomes) issus du célèbre dataset BraTS 2021. Il intègre un pipeline de prétraitement lourd ainsi qu'une application Streamlit pour comparer les performances de modèles de fondation biomédicaux (MedSAM) en temps réel.

## 🚀 Architecture Globale du Projet

Le projet s'articule autour de deux composants majeurs :
- **Ingénierie & Prétraitement des Données (preprocessing.ipynb)** : Extraction à la volée depuis des archives TAR, filtrage statistique basé sur la charge tumorale globale, et sérialisation au format compressé .npz.
- **Prototype Web Interactif (app.py)** : Interface de démonstration médicale permettant de tester l'impact des prompts utilisateurs (boîtes, points cliqués) sur MedSAM et SAM, couplée à un moteur de rendu volumétrique 3D interactif.

## 📁 Pipeline de Prétraitement Mathématique et Volumétrique

Pour évaluer la robustesse des modèles face aux variations d'échelles, le script isole des profils tumoraux extrêmes à partir d'un échantillonnage initial de 300 patients :

* Extraction & Calcul de Charge : Lecture des fichiers .nii.gz sans décompression sur disque via tarfile et nibabel. Calcul de la somme géométrique des voxels actifs :

$$V_{\text{tumeur}} = \sum (M(x, y, z) > 0)$$

* Stratification des Cohortes : Sélection rigoureuse de 50 patients pour pousser les modèles dans leurs retranchements :

    * Top 35 : Les plus volumineuses masses tumorales (structures complexes, œdèmes massifs).

    * Bottom 15 : Les micro-lésions (détection de structures fines, pannes de prompts).

* Conteneurisation .npz : Alignement spatial des 4 modalités IRM natales (FLAIR, T1, T1ce, T2) et du masque multi-classe fusionné.

## 💻 L'Application Streamlit : Prototypage Clinique Rapide

L'interface web permet aux utilisateurs de simuler l'action d'un radiologue en manipulant des curseurs pour guider l'IA.

Fonctionnalités Clés :

* Interactive Slice-by-Slice Exploration : Navigation fluide le long de l'axe de propagation 3D du volume cérébral.

* Multi-Prompt Simulation : Génération dynamique de boîtes de délimitation ou de points d'intérêt (positifs/négatifs) superposés en temps réel sur les images médicales.

* Dual-Model Benchmark : Inférence et évaluation simultanées de MedSAM (fine-tuné sur données médicales) et SAM (modèle de vision généraliste). L'application calcule instantanément le score Dice, l'IoU, et le temps de calcul CPU/GPU.

* Rendu Mesh 3D Interactif : Extraction des surfaces anatomiques à l'aide de l'algorithme des Marching Cubes (skimage.measure.marching_cubes) et affichage de la géométrie de la tumeur par décomposition en sous-structures (Plotly Go.Mesh3d) :

| Index Label | Sous-Région Tumorale Validée               | Couleur de Rendu 3D |
| ----------- | ------------------------------------------ | ------------------- |
| Label 1     | Noyau nécrotique et non réhaussé (NCR/NET) | ⚪ Gris              |
| Label 2     | Œdème péri-tumoral (ED)                    | 🟠 Orange           |
| Label 4     | Tumeur réhaussée (ET)                      | 🔴 Rouge            |

Algorithme de propagation d'intensité : L'application intègre également une fonction native de propagation morphologique 3D (propagate) basée sur un filtre d'intervalle de percentile d'intensité calculé sur la Master Slice, stabilisé par des opérations de fermeture et de suppression de petits objets connectés.


## 🔧 Technologies Utilisées

- **PyTorch** : Framework deep learning
- **MONAI** / **Nibabel** : Manipulation d'images médicales
- **Segment Anything** : Modèle de segmentation fondation
- **OpenCV** : Traitement d'images
- **Streamlit** : Prototypage 


## 🛠️ Installation et Utilisation

### Prérequis
Assurez-vous d'avoir configuré vos chemins d'accès vers les checkpoints de poids des modèles `(medsam_vit_b.pth et sam_vit_b_01ec64.pth)` définis dans le code.

### Installation de l'environnement
```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel surLinux / macOS
source venv/bin/activate
# Activer l'environnement virtuel sur Windows (CMD)
venv\Scripts\activate.bat

# Installer les dépendances
pip install streamlit numpy matplotlib plotly scikit-image pandas nibabel pyvista napari
```

### Exécution du pipeline de données
Ouvrez et exécutez l'intégralité du notebook `preprocessing.ipynb` pour générer votre dossier de données structurées :
```text
BraTS2021_npz_50/
  ├── BraTS2021_00001_vol_145230.npz
  └── ...
```
### Lancement de l'application Web
Activez le serveur Streamlit pour démarrer l'interface de manipulation :
```bash
streamlit run app.py
```
---

**Auteur** : Borel Goudjou
**Date** : Janvier 2026
