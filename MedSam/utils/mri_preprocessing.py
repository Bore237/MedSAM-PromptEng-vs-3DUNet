import os
import tarfile
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
import shutil

class BraTSPreprocessor:

    def __init__(self,
                 crop_coords=((56, 184), (56, 184), (13, 141)),
                 num_classes=4,
                 normalise = None):
        
        self.crop_coords = crop_coords
        self.num_classes = num_classes
        self.normalize = normalise

    # ----------------------------------------------------------
    def extract_tar(self, tar_path: str, destination: str):
        """Extract .tar file into destination folder"""
        print(f"📦 Extracting: {tar_path}")
        with tarfile.open(tar_path) as tar:
            tar.extractall(destination)
        print("✅ Extraction completed")
    
    def load_tar(self, tar_path: str, slices : tuple = ()):
        if slices:
            img = nib.load("image.nii")   # pas .nii.gz
            return img.dataobj[50:150, 50:150, 20:80]
        else:
            img = nib.load(tar_path)
            return img.get_fdata('dtype=np.float32')
    
    # ----------------------------------------------------------
    def load_and_normalize(self, path):
        """Load NIfTI (.nii.gz) and normalize per-slice"""
        img = nib.load(path).get_fdata()

        # normalize along last dimension
        if self.normalize : 
            img = img / np.max(img)

        return img

    # ----------------------------------------------------------
    def crop(self, vol):
        (x1, x2), (y1, y2), (z1, z2) = self.crop_coords
        return vol[x1:x2, y1:y2, z1:z2]

    # ----------------------------------------------------------
    def preprocess_patient(self, t1ce, t2, flair, mask):
        """Stack images, crop, normalize, one-hot encode mask"""

        t1ce_v = self.load_and_normalize(t1ce)
        t2_v = self.load_and_normalize(t2)
        flair_v = self.load_and_normalize(flair)

        mask_v = nib.load(mask).get_fdata()
        mask_v = np.where(mask_v == 4, 3, mask_v)

        # stack to shape (H, W, D, 3)
        stacked = np.stack([flair_v, t1ce_v, t2_v], axis=3)

        stacked = self.crop(stacked)
        mask_v = self.crop(mask_v)

        # one-hot encode
        mask_onehot = F.one_hot(
            torch.tensor(mask_v, dtype=torch.long),
            num_classes=self.num_classes
        ).numpy()

        return stacked, mask_onehot

    # ----------------------------------------------------------
    def save_npz(self, output_path, image, mask):
        np.savez_compressed(output_path, image=image, mask=mask)
        print(f"💾 Saved: {output_path}")

    # ----------------------------------------------------------
    def process_dataset(self, root_folder, output_folder):
        """
        root_folder: extracted data folder (contains patients)
        output_folder: where .npz will be saved
        """

        os.makedirs(output_folder, exist_ok=True)

        # get patient folders
        patients = sorted([p for p in os.listdir(root_folder)])

        for p in tqdm(patients, desc="Processing BraTS patients"):
            patient_path = os.path.join(root_folder, p)

            try:
                t1ce = [f for f in os.listdir(patient_path) if "t1ce" in f][0]
                t2   = [f for f in os.listdir(patient_path) if "t2"   in f][0]
                flair= [f for f in os.listdir(patient_path) if "flair" in f][0]
                mask = [f for f in os.listdir(patient_path) if "seg"   in f][0]

                t1ce = os.path.join(patient_path, t1ce)
                t2   = os.path.join(patient_path, t2)
                flair= os.path.join(patient_path, flair)
                mask = os.path.join(patient_path, mask)

                img, msk = self.preprocess_patient(t1ce, t2, flair, mask)

                # skip if tumor < 1%
                vals, counts = np.unique(msk[..., 1:].sum(axis=-1), return_counts=True)
                if (1 - (counts[0] / counts.sum())) < 0.01:
                    continue

                out_file = os.path.join(output_folder, p + ".npz")
                self.save_npz(out_file, img, msk)

            except Exception as e:
                print(f"❌ Error for {p}: {e}")
                continue

if __name__ == "__main__":
    pre = BraTSPreprocessor()

    pre.extract_tar("dataset.tar", "dataset_extracted")
    #BraTS2023_npz/BraTS_XXX.npz
    pre.process_dataset(
        root_folder="dataset_extracted",
        output_folder="BraTS2023_npz"
    )
    np.load("BraTS_001.npz")["image"]  # (128,128,128,3)
    np.load("BraTS_001.npz")["mask"]   # (128,128,128,4)

    data = np.load("BraTS_001.npz")
    img = torch.tensor(data["image"]).permute(3,0,1,2).unsqueeze(0).float()


