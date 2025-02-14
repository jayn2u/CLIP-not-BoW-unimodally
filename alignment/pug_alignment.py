import os
import torch
import wandb
import clip
import argparse
import sys
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loading.pug import *
from learning_alignment import *
from alignment_datasets import *
from pug_alignment_utils import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings(embedding_path):
    """Load pre-computed embeddings from disk if they exist"""
    if os.path.exists(embedding_path):
        return torch.load(embedding_path, map_location=device)
    return None

def save_embeddings(embedding_path, embeddings):
    """Save embeddings to disk for future use"""
    torch.save(embeddings, embedding_path)

def run_hnb_alignment(dataset, image_embeddings, caption_embeddings, device, args):
    """Run Hard Negative Batch alignment
    
    Args:
        dataset: PUG dataset object containing image data and labels
        image_embeddings: Pre-computed CLIP image embeddings
        caption_embeddings: Pre-computed CLIP caption embeddings
        device: torch device (CPU/GPU)
        args: Command line arguments
        
    Returns:
        model: Trained alignment model
        split_data: Dictionary containing train/val/test split data
    """
    # Split dataset into train/val/test sets
    train_indices, val_indices, test_indices, train_pair_labels, val_pair_labels, test_pair_labels = \
        train_val_test_split(dataset.df, dataset.pair_labels_dict)
    
    # Extract embeddings for each data split
    train_embeddings = image_embeddings[train_indices]
    val_embeddings = image_embeddings[val_indices]
    test_embeddings = image_embeddings[test_indices]
    
    # Get positive and negative caption labels for each split
    train_labels, train_neg_labels = get_caption_labels(train_pair_labels, dataset.pair_labels_dict)
    val_labels, val_neg_labels = get_caption_labels(val_pair_labels, dataset.pair_labels_dict)
    test_labels, test_neg_labels = get_caption_labels(test_pair_labels, dataset.pair_labels_dict)
    
    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project="compositional-clip",
        name=f"pug-{args.dataset.lower()}-alignment-hnb",
        config={
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'model': args.clip_model
        }
    )
    
    # Create dataloaders for training and validation
    train_data = PUGEmbeddingsNeg(train_embeddings, caption_embeddings, train_labels, train_neg_labels)
    train_dataloader = DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True)
    val_data = PUGEmbeddingsNeg(val_embeddings, caption_embeddings, val_labels, val_neg_labels)
    val_dataloader = DataLoader(val_data, batch_size=wandb.config.batch_size, shuffle=False)
    
    # Initialize model and optimizer
    model = CLIPAlignment(train_embeddings.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    # Train the model using contrastive loss with negative examples
    train_model_neg(model, train_dataloader, val_dataloader, optimizer, constrastive_loss_with_negatives, device, wandb.config.epochs)
    
    return model, {
        'train': (train_embeddings, train_labels, train_neg_labels),
        'val': (val_embeddings, val_labels, val_neg_labels),
        'test': (test_embeddings, test_labels, test_neg_labels)
    }

def run_sb_alignment(dataset, image_embeddings, caption_embeddings, device, args):
    """Run Simple Batch alignment without hard negatives
    
    Similar to run_hnb_alignment but uses simple contrastive loss without explicit negatives
    """
    # Split data
    train_indices, val_indices, test_indices, train_pair_labels, val_pair_labels, test_pair_labels = \
        train_val_test_split(dataset.df, dataset.pair_labels_dict)
    
    # Get embeddings and labels
    train_embeddings = image_embeddings[train_indices]
    val_embeddings = image_embeddings[val_indices]
    test_embeddings = image_embeddings[test_indices]
    
    train_labels, train_neg_labels = get_caption_labels(train_pair_labels, dataset.pair_labels_dict)
    val_labels, val_neg_labels = get_caption_labels(val_pair_labels, dataset.pair_labels_dict)
    test_labels, test_neg_labels = get_caption_labels(test_pair_labels, dataset.pair_labels_dict)
    
    # Initialize wandb
    wandb.init(
        project="compositional-clip",
        name=f"pug-{args.dataset.lower()}-alignment-sb",
        config={
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'model': args.clip_model
        }
    )
    
    # Create dataloaders
    train_data = PUGEmbeddings(train_embeddings, caption_embeddings, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True)
    val_data = PUGEmbeddings(val_embeddings, caption_embeddings, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=wandb.config.batch_size, shuffle=False)
    
    # Initialize model and training
    model = CLIPAlignment(train_embeddings.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    # Train model
    train_model(model, train_dataloader, val_dataloader, optimizer, constrastive_loss, device, wandb.config.epochs)
    
    return model, {
        'train': (train_embeddings, train_labels, train_neg_labels),
        'val': (val_embeddings, val_labels, val_neg_labels),
        'test': (test_embeddings, test_labels, test_neg_labels)
    }

def run_ft_alignment(dataset, captions, model_clip, preprocess, device, args):
    """Run Fine-Tuning alignment by training CLIP model end-to-end
    
    Args:
        dataset: PUG dataset object
        captions: List of caption strings
        model_clip: CLIP model to fine-tune
        preprocess: CLIP preprocessing function
        device: torch device
        args: Command line arguments
    
    Returns:
        model: Fine-tuned model
        split_data: Dictionary with split data
        aligned_caption_embeddings: Aligned caption embeddings
    """
    # Split data
    train_indices, val_indices, test_indices, train_pair_labels, val_pair_labels, test_pair_labels = \
        train_val_test_split(dataset.df, dataset.pair_labels_dict)
    
    # Get filenames and labels
    train_filenames = [dataset.filenames[i] for i in train_indices]
    val_filenames = [dataset.filenames[i] for i in val_indices]
    test_filenames = [dataset.filenames[i] for i in test_indices]
    
    train_labels, train_neg_labels = get_caption_labels(train_pair_labels, dataset.pair_labels_dict)
    val_labels, val_neg_labels = get_caption_labels(val_pair_labels, dataset.pair_labels_dict)
    test_labels, test_neg_labels = get_caption_labels(test_pair_labels, dataset.pair_labels_dict)
    
    train_captions = [captions[i] for i in train_labels]
    train_neg_captions = [captions[i] for i in train_neg_labels]
    val_captions = [captions[i] for i in val_labels]
    val_neg_captions = [captions[i] for i in val_neg_labels]
    test_captions = [captions[i] for i in test_labels]
    test_neg_captions = [captions[i] for i in test_neg_labels]

    # Initialize wandb
    wandb.init(
        project="compositional-clip",
        name=f"pug-{args.dataset.lower()}-alignment-ft",
        config={
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'model': args.clip_model
        }
    )
    
    # Create datasets and dataloaders
    train_data = PUGNeg(train_filenames, train_captions, train_neg_captions, preprocessor=preprocess)
    train_dataloader = DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True)
    val_data = PUGNeg(val_filenames, val_captions, val_neg_captions, preprocessor=preprocess)
    val_dataloader = DataLoader(val_data, batch_size=wandb.config.batch_size, shuffle=False)
    test_data = PUGNeg(test_filenames, test_captions, test_neg_captions, preprocessor=preprocess)
    test_dataloader = DataLoader(test_data, batch_size=wandb.config.batch_size, shuffle=False)
    
    # Initialize model and training
    model = CLIPFTAlignment(model_clip, args.embedding_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.learning_rate)
    
    # Train model
    train_ft_model_neg(model, train_dataloader, val_dataloader, optimizer, 
                      constrastive_loss_with_negatives, device, wandb.config.epochs)
    
    # Get embeddings for evaluation
    tokenized_captions = clip.tokenize(captions).to(device)
    aligned_caption_embeddings = []
    with torch.no_grad():
        for i in range(0, len(captions), args.batch_size):
            batch_captions = tokenized_captions[i:i+args.batch_size]
            aligned_caption_embeddings.append(model.encode_text(batch_captions))
    aligned_caption_embeddings = torch.cat(aligned_caption_embeddings, dim=0)
    aligned_caption_embeddings /= aligned_caption_embeddings.norm(dim=-1, keepdim=True)
    
    # Get image embeddings
    train_dataloader = DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=False)
    train_embeddings = compute_image_embeddings(train_dataloader, model, device)
    val_embeddings = compute_image_embeddings(val_dataloader, model, device)
    test_embeddings = compute_image_embeddings(test_dataloader, model, device)

    # print(aligned_caption_embeddings.dtype, train_embeddings.dtype)
    
    return model, {
        'train': (train_embeddings, train_labels, train_neg_labels),
        'val': (val_embeddings, val_labels, val_neg_labels),
        'test': (test_embeddings, test_labels, test_neg_labels)
    }, aligned_caption_embeddings

def evaluate_results(split_data, caption_embeddings, dataset):
    """Evaluate and print results for all data splits
    
    Args:
        split_data: Dictionary containing embeddings and labels for each split
        caption_embeddings: Caption embeddings to compare against
        dataset: PUG dataset object containing label information
    """
    
    for split_name, (embeddings, labels, neg_labels) in split_data.items():
        print(f"\n{split_name.upper()}")
        similarity = embeddings @ caption_embeddings.T
        get_results_i2t(similarity, labels, dataset.pair_labels_dict, verbose=False)
        accuracy = get_accuracy(embeddings, caption_embeddings, labels, neg_labels)

def main(args):
    # Load dataset
    if args.dataset == "PUG_SPARE":
        if args.data_path is None:
            args.data_path = "../datasets/PUG_SPARE"
        dataset = PUGSPARELoader(args.data_path)
    elif args.dataset == "PUG_SPAR":
        if args.data_path is None:
            args.data_path = "../datasets/PUG_SPAR"
        dataset = PUGSPARLoader(args.data_path)
    else:
        raise ValueError("Invalid dataset. Choose from 'PUG_SPARE' or 'PUG_SPAR'.")

    # Load CLIP model
    model_clip, preprocess = clip.load(args.clip_model, device=device)

    # Get caption embeddings
    captions, caption_embeddings = make_captions(dataset.pair_labels_dict, model_clip, device)

    if args.alignment_type != "FT":
        # Load or compute image embeddings
        image_embeddings = load_embeddings(args.embedding_path)
        if image_embeddings is None:
            images = preprocess_images(dataset.filenames, preprocess)
            image_embeddings = get_image_embeddings(images, model_clip)
            save_embeddings(args.embedding_path, image_embeddings)

        # Normalize embeddings
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        caption_embeddings /= caption_embeddings.norm(dim=-1, keepdim=True)
        
        if args.alignment_type == "HNB":
            model, split_data = run_hnb_alignment(dataset, image_embeddings, caption_embeddings, device, args)
        else:  # SB
            model, split_data = run_sb_alignment(dataset, image_embeddings, caption_embeddings, device, args)

        with torch.no_grad():
            aligned_caption_embeddings = model(caption_embeddings)
    else:
        model, split_data, aligned_caption_embeddings = run_ft_alignment(
            dataset, captions, model_clip, preprocess, device, args)
        caption_embeddings = caption_embeddings.half()
    
    print("\nBefore alignment:")
    evaluate_results(split_data, caption_embeddings, dataset)

    print("\nAfter alignment:")
    evaluate_results(split_data, aligned_caption_embeddings, dataset)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["PUG_SPARE", "PUG_SPAR"], required=True)
    parser.add_argument("--alignment_type", type=str, choices=["HNB", "SB", "FT"], required=True)
    parser.add_argument("--embedding_path", type=str, default="../cache/pug_img_emb.pt")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--clip_model", type=str, default="ViT-L/14")
    
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=768)
    
    args = parser.parse_args()
    main(args)
