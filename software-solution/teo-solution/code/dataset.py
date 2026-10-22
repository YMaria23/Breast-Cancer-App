import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class MultiFolderMultimodalDataset(Dataset):

  def __init__(self, base_dir, transform=None):
    self.base_dir = base_dir
    self.transform = transform

    # 1. Load Tabular Files
    self.patient_df = pd.read_csv(
        os.path.join(base_dir, "dataset2", "patient_history_dataset.csv")
    )
    self.biomarker_df = pd.read_csv(
        os.path.join(base_dir, "dataset3", "molecular_biomarker_dataset.csv")
    )

    # 2. Merge tabular datasets on a common key (e.g., 'patient_id')
    # Adjust 'patient_id' to match the actual column name in your CSVs
    self.metadata = pd.merge(
        self.patient_df, self.biomarker_df, on="patient_id", how="inner"
    )

    # 3. Gather image file paths from dataset1 subfolders (malignant, benign, normal)
    self.samples = []
    classes = ["normal", "benign", "malignant"]
    for label_idx, cls in enumerate(classes):
      img_folder = os.path.join(base_dir, "dataset1", cls, "images")
      if os.path.exists(img_folder):
        for img_name in os.listdir(img_folder):
          if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            # Extract a matching ID or filename to link image to metadata
            self.samples.append({
                "img_path": os.path.join(img_folder, img_name),
                "filename": img_name,
                "class_label": label_idx,  # Optional multi-class label
            })

    self.samples_df = pd.DataFrame(self.samples)

  def __len__(self):
    return len(self.samples_df)

  def __getitem__(self, idx):
    row = self.samples_df.iloc[idx]

    # Load Image
    image = Image.open(row["img_path"]).convert("RGB")
    if self.transform:
      image = self.transform(image)

    # Extract numerical clinical/biomarker features for this sample
    # (Select your numeric columns from self.metadata)
    tabular_tensor = torch.tensor(
        [
            float(self.metadata.iloc[idx].get("age", 0.0)),
            float(self.metadata.iloc[idx].get("tumor_size", 0.0)),
            # Add numerical biomarker columns here as needed
        ],
        dtype=torch.float32,
    )

    # Target label (e.g., target column indicating metastasis or malignancy)
    label = torch.tensor(
        float(self.metadata.iloc[idx].get("metastasis_label", 0.0)),
        dtype=torch.float32,
    )

    return image, tabular_tensor, label