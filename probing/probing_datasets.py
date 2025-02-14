from torch.utils.data import Dataset

class ProbingTokenizedTextDataset(Dataset):
    '''
    PyTorch Dataset class for tokenized text data
    '''
    def __init__(self, captions, labels):
        self.captions = captions
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.captions[idx], self.labels[idx]
    
class ProbingImagesDataset(Dataset):
    '''
    PyTorch Dataset class for preprocessed images
    '''
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
class ProbingEmbeddingsDataset(Dataset):
    '''
    PyTorch Dataset class for image/text embeddings
    '''
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]