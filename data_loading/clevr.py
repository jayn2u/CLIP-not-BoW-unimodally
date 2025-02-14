import numpy as np
import random
import torch
import clip
from PIL import Image
from tqdm import tqdm
import os
import gdown
import zipfile
from pathlib import Path

attr_set = {'blue', 'gray', 'brown', 'cyan', 'green', 'purple', 'red', 'yellow'}
obj_set = {'cube', 'sphere', 'cylinder'}

# Create a sorted list of attributes and objects to ensure consistent indexing
attr_list = sorted(list(attr_set))
obj_list = sorted(list(obj_set))

# Create dictionaries to map attributes and objects to their corresponding indices
attr_to_idx = {attr: i for i, attr in enumerate(attr_list)}
obj_to_idx = {obj: i for i, obj in enumerate(obj_list)}
idx_to_attr = {i: attr for attr, i in attr_to_idx.items()}
idx_to_obj = {i: obj for obj, i in obj_to_idx.items()}

class CLEVRLoader:
    FILE_IDS = {
        "2obj": "1u2oCAsAOJeqR-09gwflEEvVQ75Fs0eD3",
        "3obj": "18iqYJzKIGRK_2Af3j38thc6G2CtNAs-_",
        "4obj": "1KJzsGPl0XdlyomCNADFmUnRHVFztioer",
        "5obj": "1JuiHQGXor_C86t-oNDRkaPVMvhzU9IHf",
        "6obj": "1kBIz93Tes9wCC9snr82wf0gpveKFj76k",
        "7obj": "1SVwoNrq6hw_BpfomRroSb5odwHs8yC4",
        "8obj": "18SQEliWxiCp83XKRma1Jf3rbmeGLQLjJ",
        "9obj": "1RyqFtL969H8PbuoJpWMjHBXVHi7ZNtvD",
        "10obj": "1ypTi1_fumo46lrZlWEKjXqNMtYoiNzKZ"
    }
    
    def __init__(self, dataset_path, num_objects=2, download=False):
        """
        Initialize CLEVRLoader
        
        Args:
            dataset_path (str): Path to CLEVR dataset directory
            num_objects (int): Number of objects in scenes (2 or 3)
            download (bool): Whether to download dataset if not found
        """
        self.dataset_path = Path(dataset_path)
        self.num_objects = num_objects
        self.images_dir = self.dataset_path / f"images_{num_objects}obj"
        self.labels_csv = self.dataset_path / f"output_{num_objects}obj.csv"
        
        if download:
            self._prepare_dataset()
        
        if not self.images_dir.exists() or not self.labels_csv.exists():
            raise FileNotFoundError(
                f"Dataset files not found at {self.images_dir} and {self.labels_csv}. "
                "Either provide correct paths or set download=True"
            )
        
        # Load data
        self.filenames, self.pair_labels = read_data(str(self.images_dir), str(self.labels_csv))
    
    def _prepare_dataset(self):
        """Download and prepare the dataset if it doesn't exist"""
        # Create dataset directory
        os.makedirs(self.dataset_path, exist_ok=True)
        
        config = f"{self.num_objects}obj"
        if config not in self.FILE_IDS:
            raise ValueError(f"No download links available for {self.num_objects} objects configuration")
        
        # Check if both images and CSV exist
        if not self.images_dir.exists() or not self.labels_csv.exists():
            print(f"Downloading dataset for {config}...")
            zip_path = self.dataset_path / f"clevr_{config}.zip"
            
            # Download the zip file
            gdown.download(
                f"https://drive.google.com/uc?id={self.FILE_IDS[config]}&confirm=t", 
                str(zip_path), 
                quiet=False
            )
            
            # Extract the zip file
            print(f"Extracting dataset for {config}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.dataset_path)
            
            # Clean up
            os.remove(zip_path)
            print(f"Dataset extracted to: {self.dataset_path}")

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """Get image path and caption for given index"""
        filename = self.filenames[idx]
        pair_label = self.pair_labels[idx]
        
        # Create caption from pair label
        caption = ""
        for i in range(0, len(pair_label), 2):
            caption += pair_label[i] + " " + pair_label[i+1]
            if i < len(pair_label) - 2:
                caption += " and "
                
        return filename, caption


def read_data(images_dir, labels_csv):
    '''
    Read the data from the images directory and labels csv file

    Args:
        images_dir (str): Path to the directory containing images
        labels_csv (str): Path to the CSV file containing image labels

    Returns:
        all_filenames (list): List of image filenames
        all_pair_labels (list): List of pair labels
    '''
    all_filenames = []
    all_pair_labels = []

    with open(labels_csv, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(',')
            filename = os.path.join(images_dir, data[0])

            # does the image exist?
            if not os.path.exists(filename):
                print(f"Image {filename} does not exist. Skipping.")
                continue

            pair_label = list(data[1:])

            # switch every two items, because the order is object, attribute, object, attribute, ... in the csv
            for i in range(0, len(pair_label), 2):
                pair_label[i], pair_label[i + 1] = pair_label[i + 1], pair_label[i]

            pair_label = tuple(pair_label)   # tuple of (attribute, object, attribute, object, ...)

            all_filenames.append(filename)
            all_pair_labels.append(pair_label)
    
    print("Number of images: ", len(os.listdir(images_dir)))

    return all_filenames, all_pair_labels


# Function to convert a tuple to a 3x8 matrix
def tuple_to_matrix(tup):
    '''
    Convert a tuple to a 3x8 matrix representation. The matrix has 3 rows (objects) and 8 columns (attributes). 
    Each cell in the matrix represents the count of the attribute-object pair in the tuple.
    '''
    # Initialize an empty 3x8 matrix
    matrix = np.zeros((3, 8), dtype=int)
    
    # Iterate over the tuple in pairs of (attribute, object)
    for i in range(0, len(tup), 2):
        attr = tup[i]
        obj = tup[i+1]
        
        # Find the indices for the attribute and object
        attr_idx = attr_to_idx[attr]
        obj_idx = obj_to_idx[obj]
        
        # Increment the corresponding position in the matrix
        matrix[obj_idx][attr_idx] += 1
    
    return matrix


# Function to convert a 3x8 matrix back to a tuple
def matrix_to_tuple(matrix):
    '''
    Convert a 3x8 matrix to a tuple representation: (attribute, object, attribute, object, ...).
    '''
    result = []
    
    # Iterate over the matrix
    for obj_idx in range(3):
        for attr_idx in range(8):
            count = matrix[obj_idx][attr_idx]
            if count > 0:
                # Append the attribute and object to the result tuple 'count' times
                for _ in range(count):
                    result.append(idx_to_attr[attr_idx])
                    result.append(idx_to_obj[obj_idx])
    
    return tuple(result)


def get_unique_combinations(all_pair_labels):
    '''
    Get unique combinations of attributes and objects.
    Uniqie combination is defined by unique 3x8 matrix representation of the pair labels.

    Args:
        all_pair_labels (list): List of pair labels

    Returns:
        unique_labels (list): List of unique pair labels
    '''
    unique_labels_uptoorder = list(set(all_pair_labels))
    unique_labels_uptoorder = sorted(unique_labels_uptoorder)

    matrix_labels = []

    for label in unique_labels_uptoorder:
        matrix = tuple_to_matrix(label)
        if any(np.array_equal(matrix, m) for m in matrix_labels):
            continue
        matrix_labels.append(matrix)

    unique_labels = [matrix_to_tuple(matrix) for matrix in matrix_labels]
    return unique_labels


def train_val_test_split(all_pair_labels, split_ratio=0.1, seed=42):
    """
    Split the dataset into train, validation and test sets based on unique combinations of objects and attributes.
    Ensures that similar object-attribute combinations (including reversed order) are in the same split.
    
    Args:
        all_pair_labels (list): List of pair labels (tuples of attribute-object pairs)
        split_ratio (float): Ratio of data to use for test and validation sets (each)
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: Lists of indices for (train, validation, test) splits
    """
    random.seed(seed)

    # Get unique combinations of objects and attributes, ignoring order
    unique_pair_labels = get_unique_combinations(all_pair_labels)

    # Initialize matrices to store test and validation combinations
    sampled_matrices_test = [] 
    sampled_matrices_val = []

    # Sample combinations for test set
    while len(sampled_matrices_test) < int(split_ratio * len(unique_pair_labels)):
        # Randomly sample a combination
        sample = random.sample(unique_pair_labels, 1)[0]
        matrix = tuple_to_matrix(sample)
        
        # Skip if this combination (in matrix form) is already in test set
        if any(np.array_equal(matrix, m) for m in sampled_matrices_test):
            continue 
        sampled_matrices_test.append(matrix)

        # Also add the reversed version of the combination
        # e.g., if we have "red cube and blue sphere", also add "blue sphere and red cube"
        reverse_sample = (sample[2], sample[1], sample[0], sample[3])
        matrix = tuple_to_matrix(reverse_sample)
        if any(np.array_equal(matrix, m) for m in sampled_matrices_test):
            continue
        sampled_matrices_test.append(matrix)

    # Sample combinations for validation set (similar process as test set)
    while len(sampled_matrices_val) < int(split_ratio * len(unique_pair_labels)):
        sample = random.sample(unique_pair_labels, 1)[0]
        matrix = tuple_to_matrix(sample)
        # Skip if combination is already in test or validation set
        if any(np.array_equal(matrix, m) for m in sampled_matrices_val) or any(np.array_equal(matrix, m) for m in sampled_matrices_test):
            continue
        sampled_matrices_val.append(matrix)

        # Add reversed version
        reverse_sample = (sample[2], sample[1], sample[0], sample[3])
        matrix = tuple_to_matrix(reverse_sample)
        if any(np.array_equal(matrix, m) for m in sampled_matrices_val) or any(np.array_equal(matrix, m) for m in sampled_matrices_test):
            continue
        sampled_matrices_val.append(matrix)

    # Assign each sample to train/val/test based on its matrix representation
    test_indices = []
    val_indices = []
    train_indices = []

    for i, pair_label in enumerate(all_pair_labels):
        matrix = tuple_to_matrix(pair_label)
        if any(np.array_equal(matrix, m) for m in sampled_matrices_test):
            test_indices.append(i)
        elif any(np.array_equal(matrix, m) for m in sampled_matrices_val):
            val_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, val_indices, test_indices


def target_obj_split(target_obj, all_pair_labels):
    """
    Split the dataset based on whether labels contain the target object.

    Args:
        target_obj (str): The target object to search for (e.g. 'cube', 'sphere', etc.)
        all_pair_labels (list): List of pair labels (tuples of attribute-object pairs)

    Returns:
        list: Indices of examples containing the target object
    """
    target_obj_indices = []

    for i, pair_label in enumerate(all_pair_labels):
        if target_obj in pair_label:
            target_obj_indices.append(i)

    return target_obj_indices


def pair_labels_to_attr_labels(target_obj, pair_labels):
    """
    Convert pair labels to attribute labels for a specific target object.
    
    Args:
        target_obj (str): The target object to get attributes for
        pair_labels (list): List of pair labels (tuples of attribute-object pairs)
    
    Returns:
        list: List of attribute indices corresponding to the target object
    """
    attr_labels = []

    for pair_label in pair_labels:
        if target_obj in pair_label:
            attr_labels.append(attr_to_idx[pair_label[pair_label.index(target_obj)-1]])

    return attr_labels


def preprocess_images(filenames, preprocess):
    """
    Preprocess a list of images using the given preprocessing function.
    
    Args:
        filenames (list): List of image file paths
        preprocess (callable): Preprocessing function to apply to each image
    
    Returns:
        torch.Tensor: Batch of preprocessed images
    """
    images = []
    for filename in tqdm(filenames):
        image = Image.open(filename)
        image = preprocess(image).unsqueeze(0)
        images.append(image)

    images = torch.cat(images, dim=0)
    return images


def preprocess_captions(pair_labels):
    """
    Convert pair labels to natural language captions and tokenize them.
    
    Args:
        pair_labels (list): List of pair labels (tuples of attribute-object pairs)
    
    Returns:
        torch.Tensor: Tokenized captions ready for CLIP model input
    """
    captions = []
    for pair_label in pair_labels:
        caption = ""
        for i in range(0, len(pair_label), 2):
            caption += pair_label[i] + " " + pair_label[i+1] + " and "

        caption = caption[:-5]
        captions.append(caption)

    tokenized_captions = clip.tokenize(captions)

    return tokenized_captions


def encode_images(filenames, model, preprocess, device):
    """
    Encode images using the CLIP model.
    
    Args:
        filenames (list): List of image file paths
        model: CLIP model instance
        preprocess (callable): Preprocessing function for images
        device: Device to run the model on (CPU/GPU)
    
    Returns:
        torch.Tensor: Batch of image features/embeddings
    """
    image_features = []
    for filename in tqdm(filenames):
        image = preprocess(Image.open(filename)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features.append(model.encode_image(image))

    image_features = torch.stack(image_features).float()

    return image_features


def get_text_embeddings(tokenized_captions, model, device, batch_size=512):
    """
    Get text embeddings for tokenized captions using the CLIP model.
    
    Args:
        tokenized_captions (torch.Tensor): Batch of tokenized captions
        model: CLIP model instance
        device: Device to run the model on (CPU/GPU)
        batch_size (int): Batch size for processing captions
    
    Returns:
        torch.Tensor: Text embeddings/features for the captions
    """
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_captions), batch_size)):
            text_features = model.encode_text(tokenized_captions[i:i+batch_size])
            embeddings.append(text_features)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings