import os
import torch
import wandb
import clip
import argparse
import sys
import json
import pickle
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from PIL import Image
from tqdm import tqdm

import importlib.util
import sys
import os

repo_path = "../vision-language-models-are-bows-main"
sys.path.append(repo_path)  # Add to Python path

module_name = "dataset_zoo"  # The submodule to import
module_path = os.path.join(repo_path, module_name)

spec = importlib.util.find_spec(module_name)
if spec is None:
    raise ImportError(f"Module {module_name} not found in {repo_path}")

dataset_zoo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_zoo)

# Now you can use dataset_zoo functions
from dataset_zoo.retrieval import pre_caption

from learning_alignment import *
from alignment_datasets import *
from coco_utils import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_karpathy_train_data(coco_train, coco_val, karpathy_train_ids):
    """Load training data using Karpathy split"""
    train_images = []
    train_captions = []

    for i in tqdm(range(len(coco_train))):
        if coco_train.ids[i] in karpathy_train_ids:
            image, captions = coco_train[i]
            captions = captions[0:5]
            train_images.append(image)
            train_captions.extend(captions)

    for i in tqdm(range(len(coco_val))):
        if coco_val.ids[i] in karpathy_train_ids:
            image, captions = coco_train[i]
            captions = captions[0:5]
            train_images.append(image)
            train_captions.extend(captions)
            
    return train_images, train_captions

def load_karpathy_split_data(split_file, data_dir):
    """Load validation/test data using Karpathy split"""
    split_data = json.load(open(split_file, 'r'))
    
    images = []
    captions = []
    
    for ann in tqdm(split_data):
        image = Image.open(f"{data_dir}/{ann['image']}").convert("RGB")
        images.append(image)
        captions.extend(ann["caption"][0:5])
        
    return images, captions

def load_embeddings(embedding_path):
    """Load existing embeddings or compute new ones"""
    if os.path.exists(embedding_path):
        print(f"Loading cached embeddings from {embedding_path}")
        embeddings = torch.load(embedding_path, map_location=device)
        return embeddings
    else:
        print("No embeddings found.")
        return None

def compute_embeddings(args, model, preprocess, device, images, embedding_path):
    print("Computing image embeddings...")
    image_embeddings = get_image_emb(model, preprocess, device, images, normalize=True)
    if args.save_embeddings:
        print(f"Saving image embeddings to {embedding_path}")
        torch.save(image_embeddings, embedding_path)
    return image_embeddings

def load_or_compute_captions(captions, args, pos_path, neg_path=None):
    """Load existing caption pairs or compute new ones"""
    if os.path.exists(pos_path) and (neg_path is None or os.path.exists(neg_path)):
        print(f"Loading cached captions from {pos_path}")
        with open(pos_path, 'rb') as f:
            pos_captions = pickle.load(f)
        if neg_path:
            print(f"Loading cached negative captions from {neg_path}")
            with open(neg_path, 'rb') as f:
                neg_captions = pickle.load(f)
            return pos_captions, neg_captions
        return pos_captions, None
    
    if args.alignment_type == "HNB":
        print("Computing positive and negative captions...")
        pos_captions, neg_captions = get_pos_neg_captions(captions)
        if args.save_captions:
            print(f"Saving captions to {pos_path} and {neg_path}")
            with open(pos_path, 'wb') as f:
                pickle.dump(pos_captions, f)
            with open(neg_path, 'wb') as f:
                pickle.dump(neg_captions, f)
        return pos_captions, neg_captions
    else:
        print("Processing captions...")
        pos_captions = [pre_caption(caption, max_words=40) for caption in tqdm(captions)]
        if args.save_captions:
            print(f"Saving captions to {pos_path}")
            with open(pos_path, 'wb') as f:
                pickle.dump(pos_captions, f)
        return pos_captions, None

def compute_text_embeddings(model, captions, args, embedding_path, normalize=True):
    print("Computing text embeddings...")
    text_embeddings = get_text_emb(model, captions, normalize=normalize, device=device)
    if args.save_embeddings:
        print(f"Saving text embeddings to {embedding_path}")
        torch.save(text_embeddings, embedding_path)
    return text_embeddings

def run_hnb_alignment(image_embeddings, text_embeddings, neg_text_embeddings, val_image_embeddings, val_text_embeddings, val_neg_text_embeddings, device, args):
    """Run Hard Negative Batch alignment"""
    # Initialize wandb
    wandb.init(
        project="compositional-clip",
        name=f"coco-alignment-hnb",
        config={
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'model': args.clip_model
        }
    )
    
    # Create dataloaders
    train_data = COCONegEmbeddings(image_embeddings, text_embeddings, neg_text_embeddings)
    train_dataloader = DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True)
    val_data = COCONegEmbeddings(val_image_embeddings, val_text_embeddings, val_neg_text_embeddings)
    val_dataloader = DataLoader(val_data, batch_size=wandb.config.batch_size, shuffle=False)
    
    # Initialize model and training components
    model = CLIPAlignment(image_embeddings.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    # Train model
    train_model_neg(model, train_dataloader, val_dataloader, optimizer, constrastive_loss_with_negatives, device, wandb.config.epochs)
    
    return model

def run_sb_alignment(image_embeddings, text_embeddings, val_image_embeddings, val_text_embeddings, device, args):
    """Run Simple Batch alignment"""
    # Initialize wandb
    wandb.init(
        project="compositional-clip",
        name=f"coco-alignment-sb",
        config={
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'model': args.clip_model
        }
    )
    
    # Create dataloaders
    train_data = COCOEmbeddings(image_embeddings, text_embeddings)
    train_dataloader = DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True)
    val_data = COCOEmbeddings(val_image_embeddings, val_text_embeddings)
    val_dataloader = DataLoader(val_data, batch_size=wandb.config.batch_size, shuffle=False)
    
    # Initialize model and training
    model = CLIPAlignment(image_embeddings.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    # Train model
    train_model(model, train_dataloader, val_dataloader, optimizer, constrastive_loss, device, wandb.config.epochs)
    
    return model

def main(args):
    # Load CLIP model
    model_clip, preprocess = clip.load(args.clip_model, device=device)

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    if args.use_embeddings:
        image_features_train = load_embeddings(f"{args.cache_dir}/coco_train_k_image_emb.pt")
        image_features_val = load_embeddings(f"{args.cache_dir}/coco_val_k_image_emb.pt")
        image_features_test = load_embeddings(f"{args.cache_dir}/coco_test_k_image_emb.pt")

        text_features_train = load_embeddings(f"{args.cache_dir}/coco_train_k_text_emb_pos.pt")
        text_features_val = load_embeddings(f"{args.cache_dir}/coco_val_k_text_emb_pos.pt")
        text_features_test = load_embeddings(f"{args.cache_dir}/coco_test_k_text_emb_pos.pt")

        if args.alignment_type == "HNB":
            neg_text_features_train = load_embeddings(f"{args.cache_dir}/coco_train_k_text_emb_neg.pt")
            neg_text_features_val = load_embeddings(f"{args.cache_dir}/coco_val_k_text_emb_neg.pt")

    if not args.use_embeddings or image_features_train is None or image_features_val is None or image_features_test is None:
        # Load COCO dataset
        coco_train = CocoCaptions(root=f"{args.data_path}/train2014/", 
                                annFile=f"{args.data_path}/captions_train2014.json", 
                                transform=None)
        coco_val = CocoCaptions(root=f"{args.data_path}/val2014/",
                            annFile=f"{args.data_path}/captions_val2014.json",
                            transform=None)
        
        # Load Karpathy split
        karpathy_train_ids = get_image_ids(f"{args.data_path}/coco_karpathy_train.json")
        
        # Load training data
        train_images, train_captions = load_karpathy_train_data(coco_train, coco_val, karpathy_train_ids)
        
        # Load validation and test data
        val_images, val_captions = load_karpathy_split_data(
            f"{args.data_path}/coco_karpathy_val.json", args.data_path)
        test_images, test_captions = load_karpathy_split_data(
            f"{args.data_path}/coco_karpathy_test.json", args.data_path)
        
        # Get image embeddings
        image_features_train = compute_embeddings(
            args, model_clip, preprocess, device, train_images, f"{args.cache_dir}/coco_train_k_image_emb.pt")
        image_features_val = compute_embeddings(
            args, model_clip, preprocess, device, val_images, f"{args.cache_dir}/coco_val_k_image_emb.pt")
        image_features_test = compute_embeddings(
            args, model_clip, preprocess, device, test_images, f"{args.cache_dir}/coco_test_k_image_emb.pt")
        
        # Get positive and negative captions
        pos_captions_train, neg_captions_train = load_or_compute_captions(
            train_captions, args,
            f"{args.cache_dir}/coco_train_k_pos_captions.pkl",
            f"{args.cache_dir}/coco_train_k_neg_captions.pkl")
        pos_captions_val, neg_captions_val = load_or_compute_captions(
            val_captions, args,
            f"{args.cache_dir}/coco_val_k_pos_captions.pkl",
            f"{args.cache_dir}/coco_val_k_neg_captions.pkl")
        pos_captions_test, neg_captions_test = load_or_compute_captions(
            test_captions, args,
            f"{args.cache_dir}/coco_test_k_pos_captions.pkl",
            f"{args.cache_dir}/coco_test_k_neg_captions.pkl")
        
        # Get text embeddings
        text_features_train = compute_text_embeddings(model_clip, pos_captions_train, args, f"{args.cache_dir}/coco_train_k_text_emb_pos.pt")
        text_features_val = compute_text_embeddings(model_clip, pos_captions_val, args, f"{args.cache_dir}/coco_val_k_text_emb_pos.pt")
        text_features_test = compute_text_embeddings(model_clip, pos_captions_test, args, f"{args.cache_dir}/coco_test_k_text_emb_pos.pt")

        # If using HNB, compute negative text embeddings
        if args.alignment_type == "HNB":
            neg_text_features_train = compute_text_embeddings(model_clip, neg_captions_train, args, f"{args.cache_dir}/coco_train_k_text_emb_neg.pt")
            neg_text_features_val = compute_text_embeddings(model_clip, neg_captions_val, args, f"{args.cache_dir}/coco_val_k_text_emb_neg.pt")
    
    # Run alignment
    if args.alignment_type == "HNB":
        model = run_hnb_alignment(image_features_train, text_features_train, 
                                neg_text_features_train, image_features_val,
                                text_features_val, neg_text_features_val,
                                device, args)
    else:  # SB        
        model = run_sb_alignment(image_features_train, text_features_train,
                               image_features_val, text_features_val,
                               device, args)
    
    # Save model if requested
    if args.save_model:
        model_path = f"{args.cache_dir}/coco_alignment_{args.alignment_type.lower()}.pt"
        print(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
    
    # Evaluate
    print("\nBefore alignment:")
    get_results_t2i(image_features_test, text_features_test)
    
    print("\nAfter alignment:")
    with torch.no_grad():
        aligned_text_features = model(text_features_test)
        aligned_text_features = aligned_text_features / aligned_text_features.norm(dim=-1, keepdim=True)
    get_results_t2i(image_features_test, aligned_text_features)
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment_type", type=str, choices=["HNB", "SB"], required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Path to COCO dataset directory")    # COCO 2014, Karpathy split
    parser.add_argument("--cache_dir", type=str, default="../cache", help="Directory for cached files")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    
    parser.add_argument("--use_embeddings", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true", help="Save computed embeddings")
    parser.add_argument("--save_captions", action="store_true", help="Save processed captions")
    parser.add_argument("--save_model", action="store_true", help="Save trained model")
    
    args = parser.parse_args()
    main(args)
