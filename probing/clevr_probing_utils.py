import torch
import clip
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from probing_datasets import *
from probing_models import *
from data_loading.clevr import *

def probing(target_obj, all_embeddings, all_pair_labels, batch_size, num_epochs, lr, weight_decay=0, device='cuda'):
    """Performs linear probing on pre-computed embeddings for a specific target object."""

    # Filter data to only include samples containing the target object
    target_indices = target_obj_split(target_obj, all_pair_labels)
    print(f"\nThe number of samples for the object {target_obj}: {len(target_indices)}")
    pair_labels = [all_pair_labels[i] for i in target_indices]
    embeddings = all_embeddings[target_indices]

    # Create train/val/test splits
    train_indices, val_indices, test_indices = train_val_test_split(pair_labels)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}\n")

    # Split embeddings and labels according to indices
    train_embeddings = embeddings[train_indices]
    val_embeddings = embeddings[val_indices]
    test_embeddings = embeddings[test_indices]

    train_pair_labels = [pair_labels[i] for i in train_indices]
    val_pair_labels = [pair_labels[i] for i in val_indices]
    test_pair_labels = [pair_labels[i] for i in test_indices]

    # Convert object-attribute pair labels to just attribute labels for the target object
    train_labels = pair_labels_to_attr_labels(target_obj, train_pair_labels)
    val_labels = pair_labels_to_attr_labels(target_obj, val_pair_labels)
    test_labels = pair_labels_to_attr_labels(target_obj, test_pair_labels)

    # Wrap data in PyTorch datasets
    train_dataset = ProbingEmbeddingsDataset(train_embeddings, train_labels)
    val_dataset = ProbingEmbeddingsDataset(val_embeddings, val_labels)
    test_dataset = ProbingEmbeddingsDataset(test_embeddings, test_labels)

    # Create data loaders with specified batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
    # Initialize probing classifier (8 classes for CLEVR attributes)
    model = CLIPProbing(train_embeddings.shape[-1], 8).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train the probing classifier
    train_model(model, device, criterion, optimizer, train_loader, val_loader, num_epochs)

    # Evaluate on all splits
    train_accuracy = calculate_accuracy(model, device, train_loader)
    val_accuracy = calculate_accuracy(model, device, val_loader)
    test_accuracy = calculate_accuracy(model, device, test_loader)

    print(f"Training accuracy: {train_accuracy}")
    print(f"Validation accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    return train_accuracy, val_accuracy, test_accuracy


def probing_ft_text(target_obj, all_tokenized_captions, all_pair_labels, batch_size, num_epochs, lr, model_name, device='cuda'):
    """Performs probing analysis on text features using a fine-tuned CLIP text encoder."""

    # Load CLIP model for text feature extraction
    clip_model, preprocess = clip.load(model_name, device=device)

    # Filter data to only include samples containing the target object
    target_indices = target_obj_split(target_obj, all_pair_labels)
    print(f"\nThe number of samples for the object {target_obj}: {len(target_indices)}")
    captions = [all_tokenized_captions[i] for i in target_indices]
    pair_labels = [all_pair_labels[i] for i in target_indices]

    # Create train/val/test split
    train_indices, val_indices, test_indices = train_val_test_split(pair_labels)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}\n")

    # Split captions and labels according to indices
    train_captions = [captions[i] for i in train_indices]
    val_captions = [captions[i] for i in val_indices]
    test_captions = [captions[i] for i in test_indices]

    train_pair_labels = [pair_labels[i] for i in train_indices]
    val_pair_labels = [pair_labels[i] for i in val_indices]
    test_pair_labels = [pair_labels[i] for i in test_indices]

    # Convert object-attribute pair labels to just attribute labels for the target object
    train_labels = pair_labels_to_attr_labels(target_obj, train_pair_labels)
    val_labels = pair_labels_to_attr_labels(target_obj, val_pair_labels)
    test_labels = pair_labels_to_attr_labels(target_obj, test_pair_labels)

    # Wrap data in PyTorch datasets
    train_dataset = ProbingTokenizedTextDataset(train_captions, train_labels)
    val_dataset = ProbingTokenizedTextDataset(val_captions, val_labels)
    test_dataset = ProbingTokenizedTextDataset(test_captions, test_labels)

    # Create data loaders with specified batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
    # Initialize probing classifier (8 classes for CLEVR attributes)
    model = CLIPFTProbingText(clip_model, clip_model.text_projection.shape[1], 8).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train the probing classifier
    train_model(model, device, criterion, optimizer, train_loader, val_loader, num_epochs)

    # Evaluate on all splits
    train_accuracy = calculate_accuracy(model, device, train_loader)
    val_accuracy = calculate_accuracy(model, device, val_loader)
    test_accuracy = calculate_accuracy(model, device, test_loader)

    print(f"Training accuracy: {train_accuracy}")
    print(f"Validation accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    return train_accuracy, val_accuracy, test_accuracy


def probing_ft_images(target_obj, all_images, all_pair_labels, batch_size, num_epochs, lr, model_name, device='cuda'):
    """Performs probing analysis on image features using a fine-tuned CLIP image encoder."""

    # Load CLIP model for image feature extraction
    clip_model, preprocess = clip.load(model_name, device=device)

    # Filter data to only include samples containing the target object
    target_indices = target_obj_split(target_obj, all_pair_labels)
    print(f"\nThe number of samples for the object {target_obj}: {len(target_indices)}")
    images = [all_images[i] for i in target_indices]
    pair_labels = [all_pair_labels[i] for i in target_indices]

    # Create train/val/test split
    train_indices, val_indices, test_indices = train_val_test_split(pair_labels)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}\n")

    # Split images and labels according to indices
    train_images = [images[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    test_images = [images[i] for i in test_indices]

    train_pair_labels = [pair_labels[i] for i in train_indices]
    val_pair_labels = [pair_labels[i] for i in val_indices]
    test_pair_labels = [pair_labels[i] for i in test_indices]

    # Convert object-attribute pair labels to just attribute labels for the target object
    train_labels = pair_labels_to_attr_labels(target_obj, train_pair_labels)
    val_labels = pair_labels_to_attr_labels(target_obj, val_pair_labels)
    test_labels = pair_labels_to_attr_labels(target_obj, test_pair_labels)

    # Wrap data in PyTorch datasets
    train_dataset = ProbingImagesDataset(train_images, train_labels)
    val_dataset = ProbingImagesDataset(val_images, val_labels)
    test_dataset = ProbingImagesDataset(test_images, test_labels)

    # Create data loaders with specified batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
    # Initialize probing classifier (8 classes for CLEVR attributes)
    model = CLIPFTProbingImage(clip_model, clip_model.visual.output_dim, 8).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train the probing classifier
    train_model(model, device, criterion, optimizer, train_loader, val_loader, num_epochs)

    # Evaluate on all splits
    train_accuracy = calculate_accuracy(model, device, train_loader)
    val_accuracy = calculate_accuracy(model, device, val_loader)
    test_accuracy = calculate_accuracy(model, device, test_loader)

    print(f"Training accuracy: {train_accuracy}")
    print(f"Validation accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    return train_accuracy, val_accuracy, test_accuracy
