# Évaluation de la Robustesse des Prompts (MedSAM2) • Dataset BraTS

Objectif de cette phase : Analyser et comparer la sensibilité de deux variantes du modèle MedSAM2 (Modèle Original vs. Modèle Spécialisé Lésion) face à des perturbations géométriques des boîtes englobantes (bounding boxes) lors de la segmentation de gliomes (LGG et HGG).

## Objectifs Expérimentaux

L'analyse cherche à répondre à trois questions majeures :

* Résilience aux Prompts : Le modèle conserve-t-il sa précision si l'utilisateur fournit une boîte imparfaite (étirée horizontalement ou verticalement) ?

* Spécialisation vs Généralisation : Est-ce que le modèle affiné sur les lésions hépatiques (Liver Lesion) surpasse le modèle généraliste (Original) sur des IRM cérébrales ?

* Fidélité Volumétrique & Propagation : Comment la précision se dégrade-t-elle lorsque le masque se propage de la coupe de référence (Master Slice) vers le reste du volume 3D ?

## Protocole et Pipeline de Données

Le script traite un sous-ensemble du dataset BraTS 2021 (tumeurs cérébrales) selon le flux suivant :

Données d'Entrée
* Variabilité Biologique : Tumeurs classées en LGG (Bas grade) et HGG (Haut grade).

* Modalités IRM : Analyse comparative sur FLAIR, T1ce (rehaussée), et T2.

## Stratégie de Perturbation des Boîtes (Prompts)

À partir de la boîte englobante réelle (Ground Truth), 4 déformations automatiques sont générées sur la Master Slice pour simuler des imprécisions humaines :

| Type de Box                     | Description de la Déformation |
|----------------------------------|-------------------------------|
| `original_gt`                    | Boîte englobante parfaite (référence) |
| `vert_small` / `vert_medium`     | Élargissement vertical de la boîte (facteurs progressifs) |
| `horiz_small` / `horiz_medium`   | Élargissement horizontal de la boîte (facteurs progressifs) |

## Métriques de Performance & Analyse Statistique

Pour valider scientifiquement les résultats, le pipeline calcule plusieurs indicateurs clés, adossés à des tests statistiques rigoureux :

* Précision Topologique : Score Dice (Dice_Master) et Intersection over Union (IOU_Master) sur la coupe principale.
* Cohérence Volumétrique : Erreur Volumétrique Relative (RVE en % en comparant volume_tumeur et volume_tumeur_predict).
* Dérive de Propagation (Drift) : Calcul du Delta_Dice ($\text{Best\_Slice\_Dice} - \text{Mean\_Dice\_Vol}$) pour mesurer la perte de qualité lors de la propagation 3D.

### Tests Statistiques 

* IntégrésMann-Whitney U (Alternative two-sided) :
    * Comparaison des performances globales entre LGG et HGG.
    * Évaluation de l'impact de chaque type de boîte déformée par rapport à la boîte de référence (original_gt).

* Kruskal-Wallis & Tukey HSD (Post-hoc) :
    * Analyse de l'impact de la modalité IRM (T1ce vs FLAIR vs T2) sur la réussite de la segmentation.

## Structure des Résultats Générés

Le script produit et sauvegarde un fichier centralisé de résultats : /kaggle/working/results/resultats_complets_lesion.csv.

Ce fichier regroupe pour chaque configuration :

* L'identifiant du patient (Case_ID) et sa catégorie (Catégorie).
* Le modèle et la modalité IRM testés.
* Le type de prompt appliqué (TypeBox).
* L'ensemble des métriques de Dice, de volumes et les index des coupes extrêmes (Best_Surface / Worst_Surface).

## Références

- [MedSAM2 Project sur Kaggle](https://www.kaggle.com/code/borelgoudjou/medsam2-project/)

---

**Auteur** : Borel Goudjou
**Date** : Janvier 2026