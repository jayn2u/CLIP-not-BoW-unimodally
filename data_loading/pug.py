import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import zipfile
import gdown
import clip
from pathlib import Path


class BaseLoader:
    def __init__(self, dataset_path, csv_filename):
        self.dataset_path = Path(dataset_path)
        self.csv_path = os.path.join(dataset_path, csv_filename)
        self.df = pd.read_csv(os.path.join(dataset_path, csv_filename))
        self.filenames = self.get_filenames()

    def get_filenames(self):
        return [os.path.join(self.dataset_path, filename) for filename in self.df['filename']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        caption = f"{row['character_texture']} {row['character_name']} and {row['character2_texture']} {row['character2_name']}"
        return image, caption

class PUGSPARELoader(BaseLoader):
    FILE_IDS = {
        "Desert": "1JW0sApth_vaHgMrX1zhHa-L9SZVfUF1H",
        "Island": "17GuDFHt0z8nofC2TXDOxQnb1Wp9O4Ws4",
        "MountainRange": "18xaLyYq1hEEUuOe6KoSKfjvHsqvMitfA",
        "Spaceship": "1mVEg_eLWkGjeuCd3iXPbpx5MFdXgqBnu",
    }

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self._prepare_dataset()
        super().__init__(dataset_path, "PUG_SPARE.csv")

        self.objects = self.df['character_name'].unique()
        assert len(self.objects) == 12
        self.attributes = self.df['character_texture'].unique()
        assert len(self.attributes) == 8
        self.attr_to_label = {attr: i for i, attr in enumerate(self.attributes)}
        self.pair_labels_dict = self.get_pair_labels()
        self.filenames = self.get_filenames()

    def _prepare_dataset(self):
        # Check if all required files and folders exist
        csv_path = self.dataset_path / "PUG_SPARE.csv"
        if csv_path.exists() and all(
            (self.dataset_path / env).exists() for env in self.FILE_IDS
        ):
            print("Dataset already exists. Skipping download.")
            return

        os.makedirs(self.dataset_path, exist_ok=True)

        # Download CSV if it doesn't exist
        if not csv_path.exists():
            print("Downloading CSV file...")
            gdown.download(f"https://drive.google.com/uc?id=1lGkG51-wKRhiIb__NkftvOud9QXEc41e", str(csv_path), quiet=False)
        # Download and extract each environment dataset
        for env, file_id in self.FILE_IDS.items():
            env_path = self.dataset_path / env
            if env_path.exists():
                print(f"{env} dataset already exists. Skipping download.")
                continue

            print(f"Downloading {env} dataset...")
            zip_path = self.dataset_path / f"{env}.zip"
            # url = f"https://drive.google.com/u/0/uc?id={file_id}&export=download&confirm=pbef"
            url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
            # url = f"https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}"
            gdown.download(url, str(zip_path), quiet=False, fuzzy=True)

            print(f"Extracting {env} dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.dataset_path)

            os.remove(zip_path)
            print(f"{env} dataset extracted to: {self.dataset_path}")

    def get_pair_labels(self):
        pair_labels = [
            (attr1, obj1, attr2, obj2)
            for attr1 in self.attributes
            for obj1 in self.objects
            for attr2 in self.attributes
            for obj2 in self.objects
            if obj1 != obj2 and attr1 != attr2
        ]
        return {pair_label: i for i, pair_label in enumerate(pair_labels)}

    def get_filenames(self):
        return [
            os.path.join(self.dataset_path, row['world_name'], row['filename'])
            for _, row in self.df.iterrows()
        ]


class PUGSPARLoader(BaseLoader):
    def __init__(self, dataset_path):
        super().__init__(dataset_path, "labels.csv")
        self.df = self.df[
            (self.df['character2_name'] != 'blank') &
            (self.df['character_name'] != 'blank') &
            (self.df['character_name'] != self.df['character2_name']) &
            (self.df['character_texture'] != 'Default') &
            (self.df['character2_texture'] != 'Default')
        ].reset_index(drop=True)
        self.objects = self.df['character_name'].unique()
        assert len(self.objects) == 32
        self.attributes = ['Red', 'Blue', 'Grass', 'Stone']
        self.attr_to_label = {attr: i for i, attr in enumerate(self.attributes)}
        self.pair_labels_dict = self.get_pair_labels()
        self.filenames = self.get_filenames()

    def get_pair_labels(self):
        pair_labels = []
        for obj1 in self.objects:
            for obj2 in self.objects:
                if obj1 != obj2:
                    pair_labels.append(('Blue', obj1, 'Red', obj2))
                    pair_labels.append(('Grass', obj1, 'Stone', obj2))
        return {pair_label: i for i, pair_label in enumerate(pair_labels)}

    def get_filenames(self):
        return [os.path.join(self.dataset_path, filename) for filename in self.df['filename']]



def preprocess_images(filenames, preprocess, flip=False):
    images = []
    for filename in tqdm(filenames):
        image = Image.open(filename)
        if flip and torch.rand(1).item() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        images.append(preprocess(image).unsqueeze(0))
    return torch.cat(images, dim=0)


def preprocess_captions(df):
    captions = [
        f"{row['character_texture']} {row['character_name']} and {row['character2_texture']} {row['character2_name']}"
        for _, row in df.iterrows()
    ]
    return clip.tokenize(captions)


def get_image_embeddings(images, model, batch_size=512, device='cuda'):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size)):
            embeddings.append(model.encode_image(images[i:i+batch_size].to(device)))
    return torch.cat(embeddings, dim=0).float()


def get_text_embeddings(tokenized_captions, model, batch_size=512):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_captions), batch_size)):
            embeddings.append(model.encode_text(tokenized_captions[i:i+batch_size]))
    return torch.cat(embeddings, dim=0).float()
