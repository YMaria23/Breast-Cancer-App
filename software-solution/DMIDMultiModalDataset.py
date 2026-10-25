import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer


class DMIDMultimodalDataset(Dataset):
    def __init__(self, metadata_path, image_dir, reports_dir, max_length=512):
        col_names = [
            'Image_ID', 'View', 'Tissue', 'Abnormality_Type',
            'Class', 'Coord_X', 'Coord_Y', 'Radius'
        ]

        df = pd.read_excel(metadata_path, skiprows=31, header=None, names=col_names)

        # Keep only Benign/Malignant
        df['Class'] = df['Class'].astype(str).str.strip().str.upper()
        df = df[df['Class'].isin(['B', 'M'])]

        # filtering out missing images
        df['Expected_Img_Path'] = df['Image_ID'].apply(
            lambda x: os.path.join(image_dir, f"{str(x).strip()}.png")
        )

        df = df[df['Expected_Img_Path'].apply(os.path.exists)]

        self.metadata = df.reset_index(drop=True)
        print(f"Dataset loaded: Found {len(self.metadata)} valid patients with images.")


        self.image_dir = image_dir
        self.reports_dir = reports_dir
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_length = max_length

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = str(row['Image_ID']).strip()  # ex: 'IMG001'

        # loading the image
        img_name = f"{image_id}.png"
        img_path = os.path.join(self.image_dir, img_name)

        image_id_number = str(row['Image_ID']).strip().replace('IMG', '')

        image = Image.open(img_path).convert('RGB')
        image_tensor = self.image_transform(image)

        # loading the medical file
        report_name = f"Img{image_id_number}.txt"
        report_path = os.path.join(self.reports_dir, report_name)

        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            text = "Clinical report not available."

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 0 = Benign, 1 = Malignant
        class_label = row['Class']
        label = 1 if class_label == 'M' else 0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'image': image_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label_tensor
        }