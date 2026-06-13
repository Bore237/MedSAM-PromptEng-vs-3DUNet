import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

class MedSamMetrics:
    """
    Classe utilitaire pour évaluer les masques segmentés par MedSAM
    par rapport à un masque de référence et visualiser les résultats.
    """

    def __init__(self, img_slice=None, mask_ref=None, masks_sam=None, scores=None):
        """
        img_slice : image IRM 2D utilisée comme support
        mask_ref  : masque de référence (labeled ground truth)
        masks_sam : liste/tableau des masques prédits par SAM
        scores    : scores de SAM
        """
        self.img_slice = img_slice
        self.mask_ref = mask_ref
        self.masks_sam = masks_sam
        self.scores = scores

    def dice(self, idx_mask=None):
        """
        Calcule le Dice score entre :
        - le masque de référence
        - un masque SAM choisir or si None le mask avec meilleur score
        
        Dice = 2 * |intersection| / (|ref| + |sam|)
        """
        if idx_mask == None:
            idx_mask = np.argmax(self.scores)
        mask_ref = (self.mask_ref > 0).astype(bool)
        mask_sam = (self.masks_sam[idx_mask] > 0).astype(bool)

        # Calcul des zones en commun et des tailles
        intersection = np.logical_and(mask_ref, mask_sam).sum()
        size_ref = mask_ref.sum()
        size_sam = mask_sam.sum()

        # Cas particulier : les deux masques sont vides
        if size_ref + size_sam == 0:
            return 1.0

        return 2.0 * intersection / (size_ref + size_sam)

    def iou(self, idx_mask=None):
        """
        Calcule l'Intersection over Union (IoU) entre :
        - masque de référence
        - masque SAM choisi or si None le mask avec meilleur score
        
        IoU = |intersection| / |union|
        """
        if idx_mask == None:
            idx_mask = np.argmax(self.scores)
        mask_ref = (self.mask_ref > 0).astype(bool)
        mask_sam = (self.masks_sam[idx_mask] > 0).astype(bool)

        intersection = np.logical_and(mask_ref, mask_sam).sum()
        union = np.logical_or(mask_ref, mask_sam).sum()

        # Cas particulier : les deux masques sont vides
        if union == 0:
            return 1.0

        return intersection / union

    def plot_masks(self, idx_mask=0, modality="t1ce"):
        """
        Affiche :
        - masque de référence
        - masque SAM choisi
        - l'image IRM
        - superposition des masques
        """

        if idx_mask is None:
            idx_mask = np.argmax(self.scores)

        plt.figure(figsize=(6, 4), dpi=150) 
        plt.suptitle(f"Résultat MedSAM – IRM : {modality}", fontsize=14)

        # --- Affichage IRM normalisé ---
        vmin, vmax = np.min(self.img_slice), np.max(self.img_slice)

        # Masque référence
        plt.subplot(2, 2, 1)
        plt.imshow(self.mask_ref, cmap="gray", interpolation="none")
        plt.title("Masque de référence")
        plt.axis("off")

        # Masque SAM
        plt.subplot(2, 2, 2)
        plt.imshow(self.masks_sam[idx_mask], cmap="gray", interpolation="none")
        plt.title(f"Masque SAM ({idx_mask}) – Score : {self.scores[idx_mask]:.3f}")
        plt.axis("off")

        # Image brute
        plt.subplot(2, 2, 3)
        plt.imshow(self.img_slice, cmap="gray",
                vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.title("Image IRM")
        plt.axis("off")

        # Superposition
        plt.subplot(2, 2, 4)
        plt.imshow(self.img_slice, cmap="gray",
                vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.imshow(self.masks_sam[idx_mask], alpha=0.6, cmap='Reds', interpolation="none")
        plt.imshow(self.mask_ref, alpha=0.3, cmap='Blues', interpolation="none")
        plt.title("Superposition SAM + Référence")
        plt.axis("off")

        plt.tight_layout()
        plt.show()



if __name__ == '__main':
    pass
    #utils = MedImageUtils(img_slice, mask_ref, masks_sam)
