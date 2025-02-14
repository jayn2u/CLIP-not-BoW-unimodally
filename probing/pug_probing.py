import os
import torch
import numpy as np
import clip
import argparse
import sys

# Add parent directory to system path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pug_probing_utils import *
from data_loading.pug import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings(embedding_path):
    """Load pre-computed embeddings from disk if they exist"""
    if os.path.exists(embedding_path):
        return torch.load(embedding_path, map_location=device)
    return None

def save_embeddings(embedding_path, embeddings):
    """Save computed embeddings to disk for future use"""
    torch.save(embeddings, embedding_path)

def main(args):
    # Initialize appropriate dataset loader based on command line arguments
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

    # Load CLIP model and its preprocessing function
    model, preprocess = clip.load(args.clip_model, device=device)
    
    if not args.finetune:
        # Linear probing mode (frozen CLIP embeddings)
        if args.probe_type == "image":
            # Try to load pre-computed image embeddings, compute if not found
            image_embeddings = load_embeddings(args.embedding_path)
            
            if image_embeddings is None:
                images = preprocess_images(dataset.filenames, preprocess)
                image_embeddings = get_image_embeddings(images, model, device=device)
                save_embeddings(args.embedding_path, image_embeddings)
            
            # Probe for each object type and compute average metrics
            avg_train_acc, avg_val_acc, avg_test_acc = 0, 0, 0
            for target_obj in dataset.objects:
                train_acc, val_acc, test_acc = probing(
                    target_obj, image_embeddings, dataset.df, dataset.attr_to_label,
                    args.batch_size, args.epochs, args.lr, device=device
                )
                avg_train_acc += train_acc
                avg_val_acc += val_acc
                avg_test_acc += test_acc
            # Calculate average accuracy across all objects
            avg_train_acc /= len(dataset.objects)
            avg_val_acc /= len(dataset.objects)
            avg_test_acc /= len(dataset.objects)
            print(f"Average: Train {avg_train_acc:.4f}, Val {avg_val_acc:.4f}, Test {avg_test_acc:.4f}")
        
        elif args.probe_type == "text":
            # Filter dataset to include non-repeating combinations (ignore environments and positions)
            if args.dataset == "PUG_SPARE":
                dataset.df = dataset.df[(dataset.df['world_name'] == 'Desert') & (dataset.df['character_pos'].isnull())].reset_index(drop=True)
            else:  # PUG_SPAR
                dataset.df = dataset.df[dataset.df['world_name'] == 'desert'].reset_index(drop=True)
            
            # Process captions and get text embeddings
            tokenized_captions = preprocess_captions(dataset.df).to(device)
            text_embeddings = get_text_embeddings(tokenized_captions, model)
            
            # Probe for each object type and compute average metrics
            avg_train_acc, avg_val_acc, avg_test_acc = 0, 0, 0
            for target_obj in dataset.objects:
                train_acc, val_acc, test_acc = probing(
                    target_obj, text_embeddings, dataset.df, dataset.attr_to_label,
                    args.batch_size, args.epochs, args.lr, device=device
                )
                avg_train_acc += train_acc
                avg_val_acc += val_acc
                avg_test_acc += test_acc
            avg_train_acc /= len(dataset.objects)
            avg_val_acc /= len(dataset.objects)
            avg_test_acc /= len(dataset.objects)
            print(f"Average: Train {avg_train_acc:.4f}, Val {avg_val_acc:.4f}, Test {avg_test_acc:.4f}")
    
    else:  # Fine-tuning mode
        if args.probe_type == "image":
            # Preprocess images for fine-tuning
            images = preprocess_images(dataset.filenames, preprocess)

            # Fine-tune and probe for each object type
            avg_train_acc, avg_val_acc, avg_test_acc = 0, 0, 0
            for target_obj in dataset.objects:
                train_acc, val_acc, test_acc = probing_ft_images(
                    target_obj, images, dataset.df, dataset.attr_to_label,
                    args.batch_size, args.epochs, args.lr, device=device
                )
                avg_train_acc += train_acc
                avg_val_acc += val_acc
                avg_test_acc += test_acc
            avg_train_acc /= len(dataset.objects)
            avg_val_acc /= len(dataset.objects)
            avg_test_acc /= len(dataset.objects)
            print(f"Average: Train {avg_train_acc:.4f}, Val {avg_val_acc:.4f}, Test {avg_test_acc:.4f}")
        
        elif args.probe_type == "text":
            # Filter dataset similar to linear probing
            if args.dataset == "PUG_SPARE":
                dataset.df = dataset.df[(dataset.df['world_name'] == 'Desert') & (dataset.df['character_pos'].isnull())].reset_index(drop=True)
            else:  # PUG_SPAR
                dataset.df = dataset.df[dataset.df['world_name'] == 'desert'].reset_index(drop=True)
            
            # Preprocess captions for fine-tuning
            tokenized_captions = preprocess_captions(dataset.df)
            
            # Fine-tune and probe for each object type
            avg_train_acc, avg_val_acc, avg_test_acc = 0, 0, 0
            for target_obj in dataset.objects:
                train_acc, val_acc, test_acc = probing_ft_text(
                    target_obj, tokenized_captions, dataset.df, dataset.attr_to_label,
                    args.batch_size, args.epochs, args.lr, device=device
                )
                avg_train_acc += train_acc
                avg_val_acc += val_acc
                avg_test_acc += test_acc
            avg_train_acc /= len(dataset.objects)
            avg_val_acc /= len(dataset.objects)
            avg_test_acc /= len(dataset.objects)
            print(f"Average: Train {avg_train_acc:.4f}, Val {avg_val_acc:.4f}, Test {avg_test_acc:.4f}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["PUG_SPARE", "PUG_SPAR"], required=True,
                       help="Dataset to use for probing")
    parser.add_argument("--probe_type", type=str, choices=["image", "text"], required=True,
                       help="Whether to probe image or text embeddings")
    parser.add_argument("--finetune", action="store_true",
                       help="Whether to fine-tune the model instead of linear probing")
    parser.add_argument("--embedding_path", type=str, default="../cache/pug_img_emb.pt",
                       help="Path to save/load pre-computed embeddings")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to dataset directory")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14",
                       help="CLIP model variant to use")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-1,
                       help="Learning rate for training")
    args = parser.parse_args()

    main(args)
