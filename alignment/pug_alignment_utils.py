import torch
from tqdm import tqdm
import clip
import random

def extend_test_indices(unique_pair_labels, df):
    """
    Extends test indices by finding all related combinations for each unique pair label.
    This keeps related examples in the same split.
    """
    extended_indices = []

    for unique_pair_label in unique_pair_labels:
        character_texture = unique_pair_label[0]
        character_name = unique_pair_label[1]
        character2_texture = unique_pair_label[2]
        character2_name = unique_pair_label[3]

        # Find rows with same characters and textures in different worlds
        # This maintains consistency by including all world variations
        alt_row = df[df['character_name'] == character_name]
        alt_row = alt_row[alt_row['character2_name'] == character2_name]
        alt_row = alt_row[alt_row['character_texture'] == character_texture]
        alt_row = alt_row[alt_row['character2_texture'] == character2_texture]
        extended_indices.extend(list(alt_row.index))

        # Find rows where characters are in opposite positions but same textures
        # e.g. if original has (dog:red, cat:blue), find (cat:blue, dog:red)
        swap_row = df[df['character_name'] == character2_name]
        swap_row = swap_row[swap_row['character2_name'] == character_name]
        swap_row = swap_row[swap_row['character_texture'] == character2_texture]
        swap_row = swap_row[swap_row['character2_texture'] == character_texture]
        extended_indices.extend(list(swap_row.index))

        # Find rows where textures are swapped between characters
        # e.g. if original has (dog:red, cat:blue), find (dog:blue, cat:red)
        swap_texture_row = df[df['character_name'] == character_name]
        swap_texture_row = swap_texture_row[swap_texture_row['character2_name'] == character2_name]
        swap_texture_row = swap_texture_row[swap_texture_row['character_texture'] == character2_texture]
        swap_texture_row = swap_texture_row[swap_texture_row['character2_texture'] == character_texture]
        extended_indices.extend(list(swap_texture_row.index))

        # Find rows with both swapped positions and swapped textures
        # e.g. if original has (dog:red, cat:blue), find (cat:red, dog:blue)
        swap_texture_row = df[df['character_name'] == character2_name]
        swap_texture_row = swap_texture_row[swap_texture_row['character2_name'] == character_name]
        swap_texture_row = swap_texture_row[swap_texture_row['character_texture'] == character_texture]
        swap_texture_row = swap_texture_row[swap_texture_row['character2_texture'] == character2_texture]
        extended_indices.extend(list(swap_texture_row.index))

    return extended_indices


def train_val_test_split(df, pair_labels_dict, seed=42):
    """
    Splits dataset into train/val/test while maintaining related combinations together.
    Uses 10% each for validation and test sets, with remaining 80% for training.
    
    Args:
        df: DataFrame containing the dataset
        pair_labels_dict: Dictionary mapping pair labels to indices
        seed: Random seed for reproducibility
    
    Returns:
        train_indices, val_indices, test_indices: Indices for each split
        train_labels, val_labels, test_labels: attribute-object pair labels for each split
    """
    random.seed(seed)

    # Get all unique pair combinations
    pair_labels = list(pair_labels_dict.keys())

    # Filter out reversed combinations to avoid duplicates in initial selection
    unique_combinations = []
    for pair_label in pair_labels:
        attr1, obj1, attr2, obj2 = pair_label
        if (attr1, obj2, attr2, obj1) in unique_combinations:
            continue
        if (attr2, obj1, attr1, obj2) in unique_combinations:
            continue
        unique_combinations.append(pair_label)

    # Select 10% for testing and extend with related combinations
    test_labels = random.sample(unique_combinations, int(0.1*len(unique_combinations)))
    test_indices = extend_test_indices(test_labels, df)

    # Select 10% for validation from remaining combinations
    val_labels = random.sample([combo for combo in unique_combinations if combo not in test_labels], 
                             int(0.1*len(unique_combinations)))
    val_indices = extend_test_indices(val_labels, df)

    # Use remaining data for training
    train_indices = [i for i in df.index if i not in test_indices and i not in val_indices]

    print(len(test_indices), len(val_indices), len(train_indices))

    # Extract character-texture pair labels for each split
    train_labels = []
    for i in range(len(train_indices)):
        row = df.iloc[train_indices[i]]
        train_labels.append((row['character_texture'], row['character_name'], 
                           row['character2_texture'], row['character2_name']))
 
    val_labels = []
    for i in range(len(val_indices)):
        row = df.iloc[val_indices[i]]
        val_labels.append((row['character_texture'], row['character_name'], 
                          row['character2_texture'], row['character2_name']))

    test_labels = []
    for i in range(len(test_indices)):
        row = df.iloc[test_indices[i]]
        test_labels.append((row['character_texture'], row['character_name'], 
                          row['character2_texture'], row['character2_name']))

    return train_indices, val_indices, test_indices, train_labels, val_labels, test_labels


def get_caption_labels(pair_labels, pair_label_dict):
    """
    Converts pair labels to indices and generates negative pairs for contrastive learning.
    For each positive pair, creates a negative pair by swapping attributes between characters.
    
    Args:
        pair_labels: List of character-texture pairs
        pair_label_dict: Dictionary mapping pairs to indices
    
    Returns:
        labels: Indices for positive pairs
        neg_labels: Indices for corresponding negative pairs
    """
    labels = []
    neg_labels = []

    for pair_label in pair_labels:
        # Get index for positive pair, checking both orderings
        if tuple(pair_label) in pair_label_dict:
            labels.append(pair_label_dict[tuple(pair_label)])
        elif (pair_label[2], pair_label[3], pair_label[0], pair_label[1]) in pair_label_dict:
            labels.append(pair_label_dict[(pair_label[2], pair_label[3], pair_label[0], pair_label[1])])
        else:
            raise ValueError('The pair label is not in the pair label dictionary')

        # Create negative pair by swapping textures between characters
        neg_pair_label = (pair_label[2], pair_label[1], pair_label[0], pair_label[3])
        if tuple(neg_pair_label) in pair_label_dict:
            neg_labels.append(pair_label_dict[tuple(neg_pair_label)])
        elif (neg_pair_label[2], neg_pair_label[3], neg_pair_label[0], neg_pair_label[1]) in pair_label_dict:
            neg_labels.append(pair_label_dict[(neg_pair_label[2], neg_pair_label[3], neg_pair_label[0], neg_pair_label[1])])
        else:
            raise ValueError('The pair label is not in the pair label dictionary')

    return labels, neg_labels 

def make_captions(pair_label_dict, model, device): 
    """
    Creates captions from pair labels and computes their CLIP text embeddings.
    Processes captions in batches to manage memory usage.
    
    Args:
        pair_label_dict: Dictionary mapping pair labels to indices
        model: CLIP model for computing embeddings
        device: Device to run computations on
    
    Returns:
        captions: List of generated captions
        caption_embeddings: Normalized CLIP embeddings for captions
    """
    # Generate captions by combining attributes and objects
    captions = []
    for pair_label in pair_label_dict.keys():
        caption = f"{pair_label[0]} {pair_label[1]} and {pair_label[2]} {pair_label[3]}"
        captions.append(caption)

    # Process captions in batches of 512 to compute embeddings
    with torch.no_grad():
        caption_embeddings = []
        for i in range(0, len(captions), 512):
            caption_embeddings.append(model.encode_text(clip.tokenize(captions[i:i+512]).to(device)).float())

        caption_embeddings = torch.cat(caption_embeddings, dim=0)
        # L2 normalize embeddings
        caption_embeddings /= caption_embeddings.norm(dim=-1, keepdim=True)
        
    return captions, caption_embeddings

def compute_image_embeddings(dataloader, model, device):
    """
    Computes normalized CLIP image embeddings for all images in the dataloader.
    
    Args:
        dataloader: DataLoader containing images
        model: CLIP model for computing embeddings
        device: Device to run computations on
    
    Returns:
        image_embeddings: Normalized CLIP embeddings for all images
    """
    image_embeddings = []
    with torch.no_grad():
        for images, _, _ in tqdm(dataloader):
            images = images.to(device)
            image_embedding = model.encode_image(images)
            image_embeddings.append(image_embedding)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    # L2 normalize embeddings
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

    return image_embeddings

def get_results_i2t(similarity, labels, pair_label_dict, verbose=False):
    """
    Computes image-to-text retrieval metrics.
    """
    pred_true_1 = 0
    pred_true_5 = 0
    pred_true_10 = 0
    reverse_correct = 0

    for i in range(len(labels)):
        pred = similarity[i]
        b = pred.argsort()    # Sort predictions by similarity score
        true_index = labels[i]
        k = (b.flip(0) == true_index).nonzero().item()

        # Check if true caption is in top-k predictions
        if true_index in b[-1:]:
            pred_true_1 = pred_true_1 + 1

        if true_index in b[-5:]:
            pred_true_5 = pred_true_5 + 1

        if true_index in b[-10:]:
            pred_true_10 = pred_true_10 + 1
            
        if true_index not in b[-1:]:    # if the true label is not top prediction
            if verbose:
                print(f"True label {list(pair_label_dict.keys())[true_index]}, predicted label {list(pair_label_dict.keys())[b[-1:]]}")

            pred = list(pair_label_dict.keys())[b[-1:]]
            true = list(pair_label_dict.keys())[true_index]

            # Check if the reversed attribute label is predicted
            if (pred[2], pred[1], pred[0], pred[3]) == true or (pred[0], pred[3], pred[2], pred[1]) == true:
                reverse_correct += 1
    
    print(f"\nR@1: {pred_true_1/len(similarity)}")
    print(f"R@5: {pred_true_5/len(similarity)}")
    print(f"R@10: {pred_true_10/len(similarity)}")
    print(f"Reverse R@1: {reverse_correct/len(similarity)}")


def get_accuracy(image_embeddings, caption_embeddings, pair_indices, neg_pair_indices):
    """
    Computes contrastive accuracy by comparing positive and negative pair similarities.
    """
    # Get embeddings for positive and negative caption pairs
    pos_caption_embeddings = caption_embeddings[pair_indices]
    neg_caption_embeddings = caption_embeddings[neg_pair_indices]

    # Compute cosine similarities
    pos_similarity = torch.sum(image_embeddings * pos_caption_embeddings, dim=-1)
    neg_similarity = torch.sum(image_embeddings * neg_caption_embeddings, dim=-1)

    # Count cases where positive similarity > negative similarity
    correct = (pos_similarity > neg_similarity).sum().item()

    print(f"Accuracy: {correct/len(image_embeddings)}")

    return correct/len(image_embeddings)