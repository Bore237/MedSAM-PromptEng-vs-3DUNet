# U-Net 3D - Segmentation Médicale Volumétrique

## 📋 Description

Implémentation d'une **U-Net 3D** pour la segmentation sémantique d'images médicales volumétriques. Ce projet applique l'architecture U-Net au contexte 3D pour segmenter des structures anatomiques dans les images CT/MRI (données BraTS 2023).

## 🎯 Objectifs

- Implémenter et entraîner une U-Net 3D pour la segmentation de tumeurs cérébrales
- Développer plusieurs versions optimisées (v0, v1)
- Gérer des données volumétriques haute résolution (128³ voxels)
- Évaluer avec des métriques médicales (Dice, Sensitivity, Specificity)

## 📚 Concepts de Segmentation Maîtrisés

### Architectures Profondes
- **U-Net** : Architecture encoder-decoder avec connexions skip
- **U-Net 3D** : Extension volumétrique de la U-Net 2D
- **Convolutions 3D** : Opérations sur volumes complets
- **Batch Normalization & Dropout** : Régularisation en haute dimension

### Traitement Volumétrique
- **Données 3D** : Gestion de volumes médicaux (128×128×128 voxels)
- **Data Loading** : Pipeline efficace avec PyTorch DataLoader
- **Augmentation 3D** : Rotations, flips, elastic deformations
- **Normalisation** : Z-score normalization sur volumes

### Entraînement et Optimisation
- **Loss Functions** : Dice Loss, Cross-Entropy Loss, combinaisons hybrides
- **Optimiseurs** : Adam avec learning rate scheduling
- **Validation** : Stratégie train/val avec métriques en temps réel
- **Tensorboard** : Suivi des expériences et visualisation

### Métriques d'Évaluation Médicales
- **Dice Score** : Chevauchement prédiction/ground truth
- **IOU (Intersection over Union)** : Évaluation des régions
- **Sensitivity & Specificity** : Performance clinique
- **Hausdorff Distance** : Erreur de contour

## 🗂️ Structure du Projet

```
U_net/
├── requirements.txt                    # Dépendances du projet
├── v0_segmentation_u_net.ipynb        # Version initiale
├── segmentation_u_net _v1.ipynb       # Améliorations v1
└── 3D_UNet_Brats2023/
    ├── version_1/                     # Optimisations architecture
    │   └── tboard_logs/              # Logs TensorBoard
    └── version_2/                     # Améliorations avancées
        └── tboard_logs/              # Logs expériences
```

## 🔧 Technologies Utilisées

- **PyTorch** : Framework deep learning principal
- **segmentation-models-pytorch-3d** : Implémentations U-Net 3D optimisées
- **MONAI** / **Nibabel** : Manipulation d'images médicales
- **TorchMetrics** : Calcul des métriques
- **TensorBoard** : Visualisation d'entraînement
- **Scikit-learn** : Analyses complémentaires

## 📊 Méthodologie

### Phase 1 : Préparation des Données
```
BraTS2023 (brut) → Redimensionnement → Normalisation → 128³ voxels
Train: 70% | Val: 30%
```

### Phase 2 : Architecture Réseau
```
Input (128³)
    ↓
Encoder (convolutions 3D + max-pooling)
    ↓
Bottleneck
    ↓
Decoder (convolutions + upsampling + skip connections)
    ↓
Output (segmentation map)
```

### Phase 3 : Entraînement
- **Loss** : Combinaison Dice + Cross-Entropy
- **Batch Size** : Adapté à la mémoire GPU
- **Epochs** : 50-100 avec early stopping
- **Learning Rate** : Adaptive scheduling (cosine annealing)

### Phase 4 : Évaluation
- Calcul des métriques par slice et par volume
- Visualisation des prédictions en 3D
- Analyse des cas d'erreur

## 💡 Apprentissages Clés

✅ **Architecture U-Net** : Principes encoder-decoder et skip connections  
✅ **Deep Learning 3D** : Défis computationnels et mémoire GPU  
✅ **Loss Functions** : Choix adaptés au déséquilibre des classes  
✅ **Data Augmentation** : Stratégies pour petit datasets  
✅ **Metrics Médicales** : Interprétation clinique des résultats  
✅ **Expérimentation** : Versionning et suivi des performances (v0→v1→v2)

## 📈 Progression des Versions

| Version | Améliorations |
|---------|--------------|
| **v0** | Prototype U-Net 3D basique |
| **v1** | Optimisation architecture, meilleure augmentation |

## 📊 Résultats Attendus

- Dice Score > 0.85 sur validation set
- Généralisation sur données hors-distribution
- Inference rapide (~2-5s par volume 128³)

---

**Auteur** : Segmentation Project  
**Date** : Décembre 2025  
**Dataset** : BraTS 2023 Preprocessed (128³ resolution)
