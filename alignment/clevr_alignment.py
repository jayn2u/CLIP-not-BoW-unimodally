import os
import torch
import wandb
import clip
import argparse
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loading.clevr import *
from learning_alignment import *
from alignment_datasets import *
from clevr_alignment_utils import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_and_save_embeddings(filenames, model, preprocess, device, embedding_path):
    """Compute and save image embeddings"""
    print("Computing image embeddings...")
    images = preprocess_images(filenames, preprocess)
    image_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), 512)):
            batch = images[i:i+512].to(device)
            image_embeddings.append(model.encode_image(batch))
    
    image_embeddings = torch.cat(image_embeddings, dim=0).float()
    
    # Save embeddings
    torch.save(image_embeddings, embedding_path)
    print(f"Saved image embeddings to {embedding_path}")
    
    return image_embeddings

def load_or_compute_embeddings(filenames, model, preprocess, device, args):
    """Load existing embeddings or compute new ones"""
    if os.path.exists(args.embedding_path) and not args.recompute_embeddings:
        print(f"Loading cached image embeddings from {args.embedding_path}")
        image_embeddings = torch.load(args.embedding_path)
    else:
        image_embeddings = compute_and_save_embeddings(
            filenames, model, preprocess, device, args.embedding_path)
    
    return image_embeddings.squeeze(1)

def run_hnb_alignment(filenames, all_pair_labels, caption_embeddings, model, preprocess, device, args):
    """Run Hard Negative Batch alignment"""
    # Load or compute image embeddings
    image_embeddings = load_or_compute_embeddings(filenames, model, preprocess, device, args).float()
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    
    # Split data
    train_indices, val_indices, test_indices = train_val_test_split(all_pair_labels, 0.1)
    
    # Get embeddings for each split
    train_embeddings = image_embeddings[train_indices]
    val_embeddings = image_embeddings[val_indices] 
    test_embeddings = image_embeddings[test_indices]
    
    # Get pair labels for each split
    train_pair_labels = [all_pair_labels[i] for i in train_indices]
    val_pair_labels = [all_pair_labels[i] for i in val_indices]
    test_pair_labels = [all_pair_labels[i] for i in test_indices]
    
    # Get caption labels
    pair_label_dict = get_pair_labels(attr_set, obj_set)
    train_labels, train_neg_labels = get_caption_labels(train_pair_labels, pair_label_dict)
    val_labels, val_neg_labels = get_caption_labels(val_pair_labels, pair_label_dict)
    test_labels, test_neg_labels = get_caption_labels(test_pair_labels, pair_label_dict)
    
    # Initialize wandb
    wandb.init(
        project="compositional-clip",
        name="clevr-alignment-hnb",
        config={
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'model': args.clip_model
        }
    )
    
    # Create dataloaders
    train_data = CLEVREmbeddingsNeg(train_embeddings, caption_embeddings, train_labels, train_neg_labels)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_data = CLEVREmbeddingsNeg(val_embeddings, caption_embeddings, val_labels, val_neg_labels)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model and training
    model = CLIPAlignment(train_embeddings.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    train_model_neg(model, train_dataloader, val_dataloader, optimizer, 
                   constrastive_loss_with_negatives, device, args.epochs)
    
    return model, {
        'train': (train_embeddings, train_labels, train_neg_labels),
        'val': (val_embeddings, val_labels, val_neg_labels),
        'test': (test_embeddings, test_labels, test_neg_labels)
    }

def run_sb_alignment(filenames, all_pair_labels, caption_embeddings, model, preprocess, device, args):
    """Run Simple Batch alignment"""
    # Load or compute image embeddings
    image_embeddings = load_or_compute_embeddings(filenames, model, preprocess, device, args).float()
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    
    # Split data
    train_indices, val_indices, test_indices = train_val_test_split(all_pair_labels, 0.1)
    
    # Get embeddings for each split
    train_embeddings = image_embeddings[train_indices]
    val_embeddings = image_embeddings[val_indices]
    test_embeddings = image_embeddings[test_indices]
    
    # Get pair labels for each split
    train_pair_labels = [all_pair_labels[i] for i in train_indices]
    val_pair_labels = [all_pair_labels[i] for i in val_indices]
    test_pair_labels = [all_pair_labels[i] for i in test_indices]
    
    # Get caption labels
    pair_label_dict = get_pair_labels(attr_set, obj_set)
    train_labels, train_neg_labels = get_caption_labels(train_pair_labels, pair_label_dict)
    val_labels, val_neg_labels = get_caption_labels(val_pair_labels, pair_label_dict)
    test_labels, test_neg_labels = get_caption_labels(test_pair_labels, pair_label_dict)
    
    # Initialize wandb
    wandb.init(
        project="compositional-clip",
        name="clevr-alignment-sb",
        config={
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'model': args.clip_model
        }
    )
    
    # Create dataloaders
    train_data = CLEVREmbeddings(train_embeddings, caption_embeddings, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_data = CLEVREmbeddings(val_embeddings, caption_embeddings, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model and training
    model = CLIPAlignment(train_embeddings.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    train_model(model, train_dataloader, val_dataloader, optimizer, 
                constrastive_loss, device, args.epochs)
    
    return model, {
        'train': (train_embeddings, train_labels, train_neg_labels),
        'val': (val_embeddings, val_labels, val_neg_labels),
        'test': (test_embeddings, test_labels, test_neg_labels)
    }

def run_ft_alignment(filenames, all_pair_labels, captions, model_clip, preprocess, device, args):
    """Run Fine-Tuning alignment"""
    # Split data
    train_indices, val_indices, test_indices = train_val_test_split(all_pair_labels, 0.1)
    
    # Get pair labels and captions
    pair_label_dict = get_pair_labels(attr_set, obj_set)
    train_pair_labels = [all_pair_labels[i] for i in train_indices]
    val_pair_labels = [all_pair_labels[i] for i in val_indices]
    test_pair_labels = [all_pair_labels[i] for i in test_indices]
    
    # Get caption labels
    train_labels, train_neg_labels = get_caption_labels(train_pair_labels, pair_label_dict)
    val_labels, val_neg_labels = get_caption_labels(val_pair_labels, pair_label_dict)
    test_labels, test_neg_labels = get_caption_labels(test_pair_labels, pair_label_dict)
    
    # Get filenames and captions
    train_filenames = [filenames[i] for i in train_indices]
    val_filenames = [filenames[i] for i in val_indices]
    test_filenames = [filenames[i] for i in test_indices]
    
    train_captions = [captions[i] for i in train_labels]
    train_neg_captions = [captions[i] for i in train_neg_labels]
    val_captions = [captions[i] for i in val_labels]
    val_neg_captions = [captions[i] for i in val_neg_labels]
    test_captions = [captions[i] for i in test_labels]
    test_neg_captions = [captions[i] for i in test_neg_labels]
    
    # Initialize wandb
    wandb.init(
        project="compositional-clip",
        name="clevr-alignment-ft",
        config={
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'model': args.clip_model
        }
    )
    
    # Create dataloaders
    train_data = CLEVRNeg(train_filenames, train_captions, train_neg_captions, preprocessor=preprocess)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_data = CLEVRNeg(val_filenames, val_captions, val_neg_captions, preprocessor=preprocess)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_data = CLEVRNeg(test_filenames, test_captions, test_neg_captions, preprocessor=preprocess)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model and training
    model = CLIPFTAlignment(model_clip, args.embedding_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    
    # Train model
    train_ft_model_neg(model, train_dataloader, val_dataloader, optimizer, 
                      constrastive_loss_with_negatives, device, args.epochs)
    
    # Get aligned embeddings
    tokenized_captions = clip.tokenize(captions).to(device)
    aligned_caption_embeddings = []
    with torch.no_grad():
        for i in range(0, len(captions), args.batch_size):
            batch_captions = tokenized_captions[i:i+args.batch_size]
            aligned_caption_embeddings.append(model.encode_text(batch_captions))
    aligned_caption_embeddings = torch.cat(aligned_caption_embeddings, dim=0)
    aligned_caption_embeddings /= aligned_caption_embeddings.norm(dim=-1, keepdim=True)
    
    # Get image embeddings
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    train_embeddings = get_image_embeddings(train_dataloader, model, device)
    val_embeddings = get_image_embeddings(val_dataloader, model, device)
    test_embeddings = get_image_embeddings(test_dataloader, model, device)
    
    return model, {
        'train': (train_embeddings, train_labels, train_neg_labels),
        'val': (val_embeddings, val_labels, val_neg_labels),
        'test': (test_embeddings, test_labels, test_neg_labels)
    }, aligned_caption_embeddings

def evaluate_results(split_data, caption_embeddings, pair_label_dict):
    """Evaluate and print results for all splits"""
    for split_name, (embeddings, labels, neg_labels) in split_data.items():
        print(f"\n{split_name.upper()}")
        get_results_i2t(embeddings, caption_embeddings, labels, pair_label_dict, verbose=False)
        get_accuracy(embeddings, caption_embeddings, labels, neg_labels)

def main(args):
    # Load CLIP model
    model_clip, preprocess = clip.load(args.clip_model, device=device)
    
    # Load dataset using CLEVRLoader
    clevr_loader = CLEVRLoader(args.data_path, num_objects=2, download=args.download)
    filenames, all_pair_labels = clevr_loader.filenames, clevr_loader.pair_labels
    
    # Get captions and embeddings
    pair_label_dict = get_pair_labels(attr_set, obj_set)
    captions, caption_embeddings = get_captions(pair_label_dict, model_clip, device, normalize=False)
    
    if args.alignment_type == "HNB":
        model, split_data = run_hnb_alignment(
            filenames, all_pair_labels, caption_embeddings, model_clip, preprocess, device, args)
        with torch.no_grad():
            aligned_caption_embeddings = model(caption_embeddings)
    elif args.alignment_type == "SB":
        model, split_data = run_sb_alignment(
            filenames, all_pair_labels, caption_embeddings, model_clip, preprocess, device, args)
        with torch.no_grad():
            aligned_caption_embeddings = model(caption_embeddings)
    else:  # FT
        model, split_data, aligned_caption_embeddings = run_ft_alignment(
            filenames, all_pair_labels, captions, model_clip, preprocess, device, args)
        caption_embeddings = caption_embeddings.half()
    
    print("\nBefore alignment:")
    evaluate_results(split_data, caption_embeddings, pair_label_dict)
    
    print("\nAfter alignment:")
    evaluate_results(split_data, aligned_caption_embeddings, pair_label_dict)
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment_type", type=str, choices=["HNB", "SB", "FT"], required=True)
    parser.add_argument("--data_path", type=str, default="../datasets/CLEVR")
    parser.add_argument("--embedding_path", type=str, default="../cache/clevr_2obj_img_emb.pt")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14")
    parser.add_argument("--recompute_embeddings", action="store_true", 
                       help="Force recomputation of image embeddings")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset if not found")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=768)
    
    args = parser.parse_args()
    main(args)