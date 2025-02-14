from torch.utils.data import Dataset
from PIL import Image
import clip

class PUGEmbeddingsNeg(Dataset):
    """Handles pre-computed embeddings for images and text, along with positive and negative pair labels."""
    def __init__(self, image_embeddings, text_embeddings, pair_labels, neg_pair_labels):
        self.image_embeddings = image_embeddings  # Pre-computed image embeddings
        self.pair_labels = pair_labels  # Labels for positive image-text pairs
        self.text_embeddings = text_embeddings  # Pre-computed text embeddings
        self.neg_pair_labels = neg_pair_labels  # Labels for negative image-text pairs

    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.text_embeddings[self.pair_labels[idx]], self.text_embeddings[self.neg_pair_labels[idx]]
    
class PUGEmbeddings(Dataset):
    """Handles pre-computed embeddings for positive image-text pairs."""
    def __init__(self, image_embeddings, text_embeddings, pair_labels):
        self.image_embeddings = image_embeddings  # Pre-computed image embeddings
        self.pair_labels = pair_labels  # Labels for positive image-text pairs
        self.text_embeddings = text_embeddings  # Pre-computed text embeddings

    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.text_embeddings[self.pair_labels[idx]]
    
class PUG(Dataset):
    """Dataset class for raw PUG data.
    Processes raw images and captions using CLIP preprocessing."""
    def __init__(self, filenames, captions, preprocessor):
        self.filenames = filenames  # Paths to image files
        self.captions = captions  # Corresponding text captions
        self.preprocessor = preprocessor  # CLIP image preprocessor

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.filenames[idx])
        processed_image = self.preprocessor(image)

        # Tokenize caption using CLIP tokenizer
        tokenized_caption = clip.tokenize(self.captions[idx])

        return processed_image, tokenized_caption
    
class PUGNeg(Dataset):
    """Dataset class for raw PUG data with negative examples.
    Processes raw images and both positive and negative captions."""
    def __init__(self, filenames, captions, neg_captions, preprocessor):
        self.filenames = filenames  # Paths to image files
        self.captions = captions  # Positive text captions
        self.neg_captions = neg_captions  # Negative text captions
        self.preprocessor = preprocessor  # CLIP image preprocessor

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.filenames[idx])
        processed_image = self.preprocessor(image)

        # Tokenize both positive and negative captions
        tokenized_caption = clip.tokenize(self.captions[idx]).squeeze(0)
        neg_tokenized_caption = clip.tokenize(self.neg_captions[idx]).squeeze(0)

        return processed_image, tokenized_caption, neg_tokenized_caption
    

class CLEVREmbeddingsNeg(Dataset):
    """Handles pre-computed embeddings for images and text, along with positive and negative pair labels."""
    def __init__(self, image_embeddings, text_embeddings, pair_labels, neg_pair_labels):
        self.image_embeddings = image_embeddings  # Pre-computed image embeddings
        self.pair_labels = pair_labels  # Labels for positive image-text pairs
        self.text_embeddings = text_embeddings  # Pre-computed text embeddings
        self.neg_pair_labels = neg_pair_labels  # Labels for negative image-text pairs

    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.text_embeddings[self.pair_labels[idx]], self.text_embeddings[self.neg_pair_labels[idx]]
    

class CLEVREmbeddings(Dataset):
    """Handles pre-computed embeddings for positive image-text pairs."""
    def __init__(self, image_embeddings, text_embeddings, pair_labels):
        self.image_embeddings = image_embeddings  # Pre-computed image embeddings
        self.pair_labels = pair_labels  # Labels for positive image-text pairs
        self.text_embeddings = text_embeddings  # Pre-computed text embeddings

    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.text_embeddings[self.pair_labels[idx]]
    

class CLEVRNeg(Dataset):
    """Handles raw images and captions (both positive and negative) with CLIP preprocessing."""
    def __init__(self, filenames, captions, neg_captions, preprocessor):
        self.filenames = filenames  # Paths to image files
        self.captions = captions  # Positive text captions
        self.neg_captions = neg_captions  # Negative text captions
        self.preprocessor = preprocessor  # CLIP image preprocessor

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.filenames[idx])
        processed_image = self.preprocessor(image)

        # Tokenize both positive and negative captions
        tokenized_caption = clip.tokenize(self.captions[idx]).squeeze(0)
        neg_tokenized_caption = clip.tokenize(self.neg_captions[idx]).squeeze(0)

        return processed_image, tokenized_caption, neg_tokenized_caption
    

class COCOEmbeddings(Dataset):
    """Handles pre-computed embeddings for images and text.
    Note: Each image has 5 corresponding captions."""
    def __init__(self, image_embeddings, text_embeddings):
        self.image_embeddings = image_embeddings  # Pre-computed image embeddings
        self.text_embeddings = text_embeddings  # Pre-computed text embeddings

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        # Returns a pair of (image_embedding, text_embedding)
        # Note: idx//5 is used because each image has 5 captions
        return self.image_embeddings[idx//5], self.text_embeddings[idx]
    

class COCONegEmbeddings(Dataset):
    """Handles pre-computed embeddings for images and text, including negative pairs.
    Note: Each image has 5 corresponding captions."""
    def __init__(self, image_embeddings, text_embeddings, neg_text_embeddings):
        self.image_embeddings = image_embeddings  # Pre-computed image embeddings
        self.text_embeddings = text_embeddings  # Pre-computed text embeddings
        self.neg_text_embeddings = neg_text_embeddings  # Pre-computed negative text embeddings

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        # Returns a triplet of (image_embedding, positive_text_embedding, negative_text_embedding)
        # Note: idx//5 is used because each image has 5 captions
        return self.image_embeddings[idx//5], self.text_embeddings[idx], self.neg_text_embeddings[idx]