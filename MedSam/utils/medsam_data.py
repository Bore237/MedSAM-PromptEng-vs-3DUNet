import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class MRIPreprocessingImage:
    def __init__(self, dataset, n_samples=10, slice_idx=64):
        """
        dataset : PyTorch Dataset (image, mask)
        n_samples : nombre d'images à utiliser (par défaut 10)
        modality : 'flair', 't1ce', 't2'
        slice_idx : index de la coupe (None = milieu)
        """
        self.dataset = dataset
        self.n_samples = n_samples
        self.modality = ["flair", "t1ce", "t2"]
        self.slice_idx = slice_idx

    def get_modalities(self, image, mask, modality = "t1ce"):
        """
        Sélectionne une modalité dans un volume multi-modal.
        Hypothèse : image.shape = [C, D, H, W]
        """
        if modality not in self.modality:
            raise ValueError("Modalité non reconnue. Choisir 'flair', 't1ce' ou 't2'.")

        idx = self.modality.index(modality)
        if modality == "flair":
            return image[idx].numpy(), 1.0 - mask[idx].numpy()
        else :
            return image[idx].numpy(), mask[idx].numpy()


    # def __getitem__(self, idx):
    def get_slice(self, vol_img, vol_mask):
        """
        Retourne une coupe 2D (slice)
        """
        if self.slice_idx is None:
            idx = vol_img.shape[0] // 2 
        else:
            idx = self.slice_idx
        return vol_img[idx,:,:], vol_mask[idx,:,:]

    def sample_data(self, modality = "t1ce"):
        """
        Sélectionne aléatoirement n_samples slices
        """
        img_slices, mask_slices = [], []

        indices = np.random.choice(len(self.dataset), size=self.n_samples, replace=False)
        for idx in indices:
            image, mask = self.dataset[idx]
            vol_img, vol_mask = self.get_modalities(image, mask, modality)
            img, msk = self.get_slice(vol_img, vol_mask)
            img_slices.append(img)
            mask_slices.append(msk)

        return img_slices, mask_slices

    def plot_samples(self, n_sample = 1):
        """
        Affiche les images et masques : 2 lignes x 3 colonnes par modalité
        """
        plt.figure(figsize=(15, 5 * n_sample))

        for j in range(n_sample):

            idx = np.random.randint(self.n_samples)

            for i, modality in enumerate(self.modality, start=1):
                imgs, masks = self.sample_data(modality)

                # Ligne 1 : images
                plt.subplot(n_sample * 2, 3, j * 6 + i)
                plt.imshow(imgs[idx], cmap="gray")
                plt.axis("off")
                plt.title(f"{modality.upper()} – Image")

                # Ligne 2 : masques
                plt.subplot(n_sample * 2, 3, j * 6 + i + 3)
                plt.imshow(masks[idx], cmap="gray")
                plt.axis("off")
                plt.title(f"{modality.upper()} – Mask")

        plt.tight_layout()
        plt.show()

class BratsDataset(Dataset):
  def __init__(self, img_dir, mask_dir, normalization=True):
    super().__init__()

    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.img_list = sorted(
        os.listdir(img_dir)
    )
    self.mask_list = sorted(os.listdir(mask_dir))
    self.normalization = normalization

    # If normalization is True, set up a normalization transform
    if self.normalization:
        self.normalizer = transforms.Normalize(
            mean=[0.5], std=[0.5]
        )  

  def load_file(self, filepath):
    return np.load(filepath)

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, idx):
    image_path = os.path.join(self.img_dir, self.img_list[idx])
    mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
    # Load the image and mask
    image = self.load_file(image_path)
    mask = self.load_file(mask_path)

    # Convert to torch tensors and permute axes to C, D, H, W format (needed for 3D models)
    image = torch.from_numpy(image).permute(3, 2, 0, 1)  # Shape: C, D, H, W
    mask = torch.from_numpy(mask).permute(3, 2, 0, 1)  # Shape: C, D, H, W

    # Normalize the image if normalization is enabled
    if self.normalization:
        image = self.normalizer(image)

    return image, mask
  

if __name__ == "__main__":
    img_dir = "D:/marchine_learning/Projet/Segmentation/BraTS2023_Preprocessed/input_data_128/train/images"
    mask_dir = "D:/marchine_learning/Projet/Segmentation/BraTS2023_Preprocessed/input_data_128/train/masks"
    
    train_dataset = BratsDataset(img_dir, mask_dir, normalization=False)

    trainer = MRIPreprocessingImage(train_dataset, n_samples=10, modality="flair", slice_idx=None)
    trainer.plot_samples()
