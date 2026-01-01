import os
import re
from fractions import Fraction
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class RecipeDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ---------------- Image ----------------
        image_name = str(row.get("Image_Name", "")).strip()
        image_path = os.path.join(self.image_dir, image_name + ".jpg")

        if image_name not in {"#NAME?", "", "None"} and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        image = self.transform(image)

        # ---------------- Text ----------------
        title = self.clean_text(row.get("Title", ""))
        ingredients = self.clean_text(row.get("Ingredients", ""))
        instructions = self.clean_text(row.get("Instructions", ""))

        return {
            "image": image,
            "input_text": title,
            "target_text": ingredients + " " + instructions,
            "metadata": {
                "title": title,
                "ingredients": ingredients,
                "instructions": instructions,
                "image_path": image_path,
            }
        }

    def clean_text(self, field):
        """
        Cleans raw recipe text safely for long documents.
        """

        invalid_values = {"", "[]", "['']", "N/A", "N/A N/A", "None"}

        def clean_item(item):
            item = str(item)

            # Replace hyphens inside words
            item = re.sub(r'(?<=\w)-(?=\w)', ' ', item)

            # Remove strange encoding artifacts
            item = re.sub(r'[¬Ω‚Äì‚Öì¬æ¬º]', '', item)

            # Keep alphanumerics + fractions
            item = re.sub(r'[^\w\s/]', ' ', item)

            # Convert fractions (e.g., 1/2 → 0.5)
            def frac_to_decimal(match):
                try:
                    return str(float(Fraction(match.group(0))))
                except Exception:
                    return match.group(0)

            item = re.sub(r'\d+/\d+', frac_to_decimal, item)

            # Normalize whitespace
            item = re.sub(r'\s+', ' ', item)

            return item.lower().strip()

        if isinstance(field, list):
            field = " ".join(
                clean_item(f) for f in field
                if str(f).strip() not in invalid_values
            )
        else:
            field = str(field).strip()
            if field in invalid_values:
                field = ""
            else:
                field = clean_item(field)

        return field if field else "[NO_TEXT]"
