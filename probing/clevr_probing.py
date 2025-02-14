import os
import torch
import clip
import argparse
import sys

# Add parent directory to Python path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loading.clevr import *
from probing.clevr_probing_utils import *

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
    # Initialize CLEVR dataset loader with 2 objects per image
    clevr_loader = CLEVRLoader(args.data_path, num_objects=2, download=args.download)
    all_filenames, all_pair_labels = clevr_loader.filenames, clevr_loader.pair_labels

    model, preprocess = clip.load(args.clip_model, device=device)

    if not args.finetune:
        # Probing without fine-tuning (using frozen CLIP embeddings)
        if args.probe_type == "image":
            # Try to load cached image embeddings
            image_embeddings = load_embeddings(args.embedding_path)
            
            if image_embeddings is None:
                # Compute and cache embeddings if not found
                print("Embeddings not found. Computing embeddings...")
                image_embeddings = encode_images(all_filenames, model, preprocess, device).squeeze(1)
                save_embeddings(args.embedding_path, image_embeddings)
            
            # Probe for each object type and compute average metrics
            avg_train_acc, avg_val_acc, avg_test_acc = 0, 0, 0
            for target_obj in ['cube', 'sphere', 'cylinder']:
                train_acc, val_acc, test_acc = probing(
                    target_obj, image_embeddings, all_pair_labels, args.batch_size, args.epochs, args.lr, device=device
                )
                avg_train_acc += train_acc
                avg_val_acc += val_acc
                avg_test_acc += test_acc
            avg_train_acc /= 3
            avg_val_acc /= 3
            avg_test_acc /= 3
            print(f"Average: Train {avg_train_acc:.4f}, Val {avg_val_acc:.4f}, Test {avg_test_acc:.4f}")
        
        elif args.probe_type == "text":
            # Get unique pair labels and compute text embeddings
            pair_labels = list(set(all_pair_labels))
            tokenized_captions = preprocess_captions(pair_labels).to(device)
            text_embeddings = get_text_embeddings(tokenized_captions, model, device).float()
            
            # Probe for each object type and compute average metrics
            avg_train_acc, avg_val_acc, avg_test_acc = 0, 0, 0
            for target_obj in ['cube', 'sphere', 'cylinder']:
                train_acc, val_acc, test_acc = probing(
                    target_obj, text_embeddings, pair_labels, args.batch_size, args.epochs, args.lr, device=device
                )
                avg_train_acc += train_acc
                avg_val_acc += val_acc
                avg_test_acc += test_acc
            # Calculate average across all object types
            avg_train_acc /= 3
            avg_val_acc /= 3
            avg_test_acc /= 3
            print(f"Average: Train {avg_train_acc:.4f}, Val {avg_val_acc:.4f}, Test {avg_test_acc:.4f}")
    
    else:  # Fine-tuning mode
        if args.probe_type == "image":
            # Preprocess all images for fine-tuning
            preprocessed_images = preprocess_images(all_filenames, preprocess)
            
            # Fine-tune and probe for each object type
            avg_train_acc, avg_val_acc, avg_test_acc = 0, 0, 0
            for target_obj in ['cube', 'sphere', 'cylinder']:
                train_acc, val_acc, test_acc = probing_ft_images(
                    target_obj, preprocessed_images, all_pair_labels, args.batch_size, args.epochs, args.lr, model_name=args.clip_model, device=device
                )
                avg_train_acc += train_acc
                avg_val_acc += val_acc
                avg_test_acc += test_acc
            avg_train_acc /= 3
            avg_val_acc /= 3
            avg_test_acc /= 3
            print(f"Average: Train {avg_train_acc:.4f}, Val {avg_val_acc:.4f}, Test {avg_test_acc:.4f}")
        
        elif args.probe_type == "text":
            # Preprocess text data for fine-tuning
            pair_labels = list(set(all_pair_labels))
            tokenized_captions = preprocess_captions(pair_labels)
            
            # Fine-tune and probe for each object type
            avg_train_acc, avg_val_acc, avg_test_acc = 0, 0, 0
            for target_obj in ['cube', 'sphere', 'cylinder']:
                train_acc, val_acc, test_acc = probing_ft_text(
                    target_obj, tokenized_captions, pair_labels, args.batch_size, args.epochs, args.lr, model_name=args.clip_model, device=device
                )
                avg_train_acc += train_acc
                avg_val_acc += val_acc
                avg_test_acc += test_acc
            avg_train_acc /= 3
            avg_val_acc /= 3
            avg_test_acc /= 3
            print(f"Average: Train {avg_train_acc:.4f}, Val {avg_val_acc:.4f}, Test {avg_test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--probe_type", type=str, choices=["image", "text"], required=True)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--embedding_path", type=str, default="../cache/clevr_2obj_img_emb.pt")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    main(args)
