import json
import torch
import clip
from tqdm import tqdm  
import sys
import os
import importlib.util

repo_path = "../vision-language-models-are-bows"
sys.path.append(repo_path)  # Add to Python path

module_name = "dataset_zoo"  # The submodule to import
module_path = os.path.join(repo_path, module_name)

spec = importlib.util.find_spec(module_name)
if spec is None:
    raise ImportError(f"Module {module_name} not found in {repo_path}")

dataset_zoo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_zoo)

from dataset_zoo.perturbations import TextShuffler
from dataset_zoo.retrieval import pre_caption

# read karpathy's dataset
def get_image_ids(file):
    """
    Extracts unique image IDs from a Karpathy split JSON file.
    
    Args:
        file: Path to JSON dataset file
    
    Returns:
        List of unique image IDs as integers
    """
    with open(file, 'r') as f:
        data = json.load(f)

    image_ids = []
    for i in range(len(data)):
        image_ids.append(int(data[i]['image_id'].split('_')[-1]))

    image_ids = list(set(image_ids))
    return image_ids

def get_pos_neg_captions(captions):
    """
    Generates positive and negative caption pairs by shuffling nouns and adjectives.
    
    Args:
        captions: List of original image captions
    
    Returns:
        Tuple of (processed_captions, shuffled_captions) where shuffled_captions
        contain rearranged nouns and adjectives from the original captions
    """
    pos_captions = []
    neg_captions = [] 
    shuffler = TextShuffler()

    for caption in tqdm(captions):
        processed_caption = pre_caption(caption, max_words=40)
        pos_captions.append(processed_caption)
        neg_caption = shuffler.shuffle_nouns_and_adj(processed_caption)
        neg_captions.append(neg_caption)
    return pos_captions, neg_captions

def get_image_emb(model, preprocess, device, images, normalize=False):
    """
    Generates CLIP image embeddings for a list of images.
    
    Args:
        model: CLIP model for encoding images
        preprocess: CLIP preprocessing transform
        device: torch device (CPU/GPU)
        images: List of input images
        normalize: Whether to L2 normalize the embeddings
    
    Returns:
        Tensor of image embeddings
    """
    model.eval()

    image_features = []

    with torch.no_grad():
        for image in tqdm(images):
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features.append(model.encode_image(image_input))

        image_features = torch.stack(image_features).squeeze().float()
        
        if normalize:
            image_features /= image_features.norm(dim=-1, keepdim=True)

        print(image_features.shape)

    return image_features

def get_text_emb(model, captions, batch_size=512, normalize=False, device='cuda'):
    """
    Generates CLIP text embeddings for a list of captions in batches.
    
    Args:
        model: CLIP model for encoding text
        captions: List of text captions
        batch_size: Number of captions to process at once
        normalize: Whether to L2 normalize the embeddings
        device: torch device (CPU/GPU)
    
    Returns:
        Tensor of text embeddings
    """
    model.eval()

    text_features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size)):
            batch = captions[i:i+batch_size]
            batch = clip.tokenize(batch).to(device)
            text_features.extend(model.encode_text(batch))

        text_features = torch.stack(text_features).float()

        if normalize:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        print(text_features.shape)

    return text_features

def get_embeddings(dataloader, model, device):
    """
    Generates both image and text embeddings for all samples in a dataloader.
    
    Args:
        dataloader: DataLoader containing (image, caption) pairs
        model: CLIP model for encoding
        device: torch device (CPU/GPU)
    
    Returns:
        Tuple of (image_embeddings, text_embeddings) tensors
    """
    model.eval()

    image_embeddings = []
    text_embeddings = []

    with torch.no_grad():
        for images, captions, _ in tqdm(dataloader):
            images = images.to(device)
            image_embedding = model.encode_image(images)
            image_embeddings.append(image_embedding)

            text_embedding = model.encode_text(captions.to(device))
            text_embeddings.append(text_embedding)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    
    return image_embeddings, text_embeddings

def get_results_t2i(image_features, text_features):
    """
    Evaluates text-to-image retrieval performance using top-k recall metrics.
    
    Args:
        image_features: Tensor of encoded images
        text_features: Tensor of encoded captions
    
    Returns:
        Tuple of (top1_recall, top5_recall, top10_recall) scores
    """
    similarity = image_features @ text_features.T
    similarity = similarity.T
    n = similarity.shape[0] 

    pred_true_1 = 0
    pred_true_5 = 0
    pred_true_10 = 0

    for i in range(n):
        pred = similarity[i]
        b = pred.argsort()
        true_index = i // 5    # each image has 5 captions

        if true_index in b[-1:]:
            pred_true_1 = pred_true_1 + 1

        if true_index in b[-5:]:
            pred_true_5 = pred_true_5 + 1

        if true_index in b[-10:]:
            pred_true_10 = pred_true_10 + 1

    print(f"\nTop 1 recall: {pred_true_1/n}")
    print(f"Top 5 recall: {pred_true_5/n}")
    print(f"Top 10 recall: {pred_true_10/n}")
    
    return pred_true_1/n, pred_true_5/n, pred_true_10/n
