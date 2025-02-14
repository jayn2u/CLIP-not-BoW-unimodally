import clip
import numpy as np
from torch.utils.data import DataLoader
from probing_datasets import *
from probing_models import *

def extend_test_indices(subset_indices, df):
    extended_indices = []

    for index in subset_indices:
        row = df.loc[index]

        # Find rows with same characters and textures in different worlds
        # This maintains consistency by including all world variations
        alt_row = df[df['character_name'] == row['character_name']]  # Match first character
        alt_row = alt_row[alt_row['character2_name'] == row['character2_name']]  # Match second character
        alt_row = alt_row[alt_row['character_texture'] == row['character_texture']]  # Match first texture
        alt_row = alt_row[alt_row['character2_texture'] == row['character2_texture']]  # Match second texture
        extended_indices.extend(list(alt_row.index))

        # Find rows where characters are in opposite positions but same textures
        # e.g. if original has (dog:red, cat:blue), find (cat:blue, dog:red)
        swap_row = df[df['character_name'] == row['character2_name']]  # Second char becomes first
        swap_row = swap_row[swap_row['character2_name'] == row['character_name']]  # First char becomes second
        swap_row = swap_row[swap_row['character_texture'] == row['character2_texture']]  # Swap textures too
        swap_row = swap_row[swap_row['character2_texture'] == row['character_texture']]
        extended_indices.extend(list(swap_row.index))

        # Find rows where textures are swapped between characters
        # e.g. if original has (dog:red, cat:blue), find (dog:blue, cat:red)
        swap_texture_row = df[df['character_name'] == row['character_name']]  # Keep same characters
        swap_texture_row = swap_texture_row[swap_texture_row['character2_name'] == row['character2_name']]
        swap_texture_row = swap_texture_row[swap_texture_row['character_texture'] == row['character2_texture']]  # Swap textures
        swap_texture_row = swap_texture_row[swap_texture_row['character2_texture'] == row['character_texture']]
        extended_indices.extend(list(swap_texture_row.index))

        # Find rows with both swapped positions and swapped textures
        # e.g. if original has (dog:red, cat:blue), find (cat:red, dog:blue)
        swap_texture_row = df[df['character_name'] == row['character2_name']]  # Swap character positions
        swap_texture_row = swap_texture_row[swap_texture_row['character2_name'] == row['character_name']]
        swap_texture_row = swap_texture_row[swap_texture_row['character_texture'] == row['character_texture']]  # Keep original textures
        swap_texture_row = swap_texture_row[swap_texture_row['character2_texture'] == row['character2_texture']]
        extended_indices.extend(list(swap_texture_row.index))

    return extended_indices

# Split data into train/val/test sets while maintaining related combinations together
def train_val_test_split(target_obj, df, seed=42):
    np.random.seed(seed)  # Set random seed for reproducibility

    # Filter to get only rows where target_obj appears as first character in desert world
    subset_target_object1 = df[df['character_name'] == target_obj]
    subset_target_object1 = subset_target_object1[(subset_target_object1['world_name'] == 'desert') | (subset_target_object1['world_name'] == 'Desert')]

    # Remove rows with position information if that column exists
    if 'character_pos' in subset_target_object1.columns:
        subset_target_object1 = subset_target_object1[subset_target_object1['character_pos'].isnull()]

    print("Number of unique combinations with the target object:", len(subset_target_object1))
    
    # Create test set (10% of base combinations)
    test_indices = []
    test_indices1 = np.random.choice(subset_target_object1.index, int(0.1*len(subset_target_object1)), replace=False)
    test_indices += extend_test_indices(test_indices1, df)  # Add related examples to test set

    # Create validation set (10% of remaining base combinations)
    val_indices = []
    remaining_indices = [i for i in subset_target_object1.index if i not in test_indices]
    val_indices1 = np.random.choice(remaining_indices, int(0.1*len(subset_target_object1)), replace=False)
    val_indices += extend_test_indices(val_indices1, df)  # Add related examples to validation set

    # Use remaining data for training
    train_indices = []
    train_indices1 = [i for i in subset_target_object1.index if i not in test_indices and i not in val_indices]
    train_indices += extend_test_indices(train_indices1, df)  # Add related examples to training set
    
    print("Train/validation/test split:", len(train_indices), len(val_indices), len(test_indices))
    return train_indices, val_indices, test_indices

# Extract attribute labels for a target object from the dataset
def get_attribute_labels(target_obj, indices, subset_2_char, attr_to_label):
    """Extract texture labels for the target object from each scene"""
    attribute_labels = []

    for i in indices:
        row = subset_2_char.iloc[i]
        # Get texture of target_obj whether it appears as first or second character
        if row['character_name'] == target_obj:
            attribute_labels.append(attr_to_label[row['character_texture']])
        elif row['character2_name'] == target_obj:
            attribute_labels.append(attr_to_label[row['character2_texture']])
        else:
            print("Error: target object not found in row")

    return attribute_labels

def probing(target_obj, embeddings, df, attr_to_label, batch_size, num_epochs, lr, weight_decay=0, device='cuda'):
    """
    Train a probing classifier to predict attributes from pre-computed embeddings
    """
    # split data and get labels
    train_indices, val_indices, test_indices = train_val_test_split(target_obj, df)

    # get the attribute (target) labels
    train_labels = get_attribute_labels(target_obj, train_indices, df, attr_to_label)
    val_labels = get_attribute_labels(target_obj, val_indices, df, attr_to_label)
    test_labels = get_attribute_labels(target_obj, test_indices, df, attr_to_label)

    train_embeddings = embeddings[train_indices]
    val_embeddings = embeddings[val_indices]
    test_embeddings = embeddings[test_indices]

    # create the datasets and dataloaders
    train_dataset = ProbingEmbeddingsDataset(train_embeddings, train_labels)
    val_dataset = ProbingEmbeddingsDataset(val_embeddings, val_labels)
    test_dataset = ProbingEmbeddingsDataset(test_embeddings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
    # define the model, optimizer and loss function
    model = CLIPProbing(train_embeddings.shape[-1], 8).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_model(model, device, criterion, optimizer, train_loader, val_loader, num_epochs)

    train_accuracy = calculate_accuracy(model, device, train_loader)
    val_accuracy = calculate_accuracy(model, device, val_loader)
    test_accuracy = calculate_accuracy(model, device, test_loader)

    print(f"{target_obj}: Train {train_accuracy:.4f}, Val {val_accuracy:.4f}, Test {test_accuracy:.4f}")
    return train_accuracy, val_accuracy, test_accuracy


def probing_ft_images(target_obj, images, df, attr_to_label, batch_size, num_epochs, lr, model_name='ViT-L/14', device='cuda'):
    """
    Fine-tune CLIP's image encoder to predict attributes
    """
    # split data and get labels
    train_indices, val_indices, test_indices = train_val_test_split(target_obj, df)

    # get the attribute (target) labels
    train_labels = get_attribute_labels(target_obj, train_indices, df, attr_to_label)
    val_labels = get_attribute_labels(target_obj, val_indices, df, attr_to_label)
    test_labels = get_attribute_labels(target_obj, test_indices, df, attr_to_label)

    train_images = [images[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    test_images = [images[i] for i in test_indices]

    # create the datasets and dataloaders
    train_dataset = ProbingImagesDataset(train_images, train_labels)
    val_dataset = ProbingImagesDataset(val_images, val_labels)
    test_dataset = ProbingImagesDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    clip_model, _ = clip.load(model_name, device=device)

    # define the model, optimizer and loss function
    model = CLIPFTProbingImage(clip_model, clip_model.visual.output_dim, 8).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_model(model, device, criterion, optimizer, train_loader, val_loader, num_epochs)

    train_accuracy = calculate_accuracy(model, device, train_loader)
    val_accuracy = calculate_accuracy(model, device, val_loader)
    test_accuracy = calculate_accuracy(model, device, test_loader)

    print(f"{target_obj}: Train {train_accuracy:.4f}, Val {val_accuracy:.4f}, Test {test_accuracy:.4f}")
    return train_accuracy, val_accuracy, test_accuracy


def probing_ft_text(target_obj, tokenized_captions, df, attr_to_label, batch_size, num_epochs, lr, model_name='ViT-L/14', device='cuda'):
    """
    Fine-tune CLIP's text encoder to predict attributes
    """
    # split data and get labels
    train_indices, val_indices, test_indices = train_val_test_split(target_obj, df)

    # get the attribute (target) labels
    train_labels = get_attribute_labels(target_obj, train_indices, df, attr_to_label)
    val_labels = get_attribute_labels(target_obj, val_indices, df, attr_to_label)
    test_labels = get_attribute_labels(target_obj, test_indices, df, attr_to_label)

    train_captions = [tokenized_captions[i] for i in train_indices]
    val_captions = [tokenized_captions[i] for i in val_indices]
    test_captions = [tokenized_captions[i] for i in test_indices]

    # create the datasets and dataloaders
    train_dataset = ProbingTokenizedTextDataset(train_captions, train_labels)
    val_dataset = ProbingTokenizedTextDataset(val_captions, val_labels)
    test_dataset = ProbingTokenizedTextDataset(test_captions, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
    clip_model, _ = clip.load(model_name, device=device)

    # define the model, optimizer and loss function
    model = CLIPFTProbingText(clip_model, 768, 8).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_model(model, device, criterion, optimizer, train_loader, val_loader, num_epochs)

    train_accuracy = calculate_accuracy(model, device, train_loader)
    val_accuracy = calculate_accuracy(model, device, val_loader)
    test_accuracy = calculate_accuracy(model, device, test_loader)

    print(f"{target_obj}: Train {train_accuracy:.4f}, Val {val_accuracy:.4f}, Test {test_accuracy:.4f}")
    return train_accuracy, val_accuracy, test_accuracy