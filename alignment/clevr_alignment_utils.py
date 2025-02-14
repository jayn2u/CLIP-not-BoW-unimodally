import torch
from tqdm import tqdm
import clip

def get_pair_labels(attr_set, obj_set):
    """
    Creates all possible combinations of attribute-object pairs, excluding duplicates.
    
    Args:
        attr_set: Set of attributes (e.g., colors, sizes)
        obj_set: Set of objects (e.g., cube, sphere)
    
    Returns:
        Dictionary mapping tuple of (attr1, obj1, attr2, obj2) to unique indices
    """
    pair_labels = []

    # make all possible combinations of the form attr1 + obj1 + attr2 + obj2, excluding pairs with the same attribute or object
    for attr1 in attr_set:
        for obj1 in obj_set:
            for attr2 in attr_set:
                for obj2 in obj_set:
                    if obj1 == obj2:    # exclude pairs with the same object
                        continue
                    if (attr1, obj1, attr2, obj2) in pair_labels:
                        continue
                    if (attr2, obj2, attr1, obj1) in pair_labels:
                        continue

                    pair_labels.append((attr1, obj1, attr2, obj2))

    # create a dictionary to map the pair labels to indices
    pair_label_dict = {pair_label: i for i, pair_label in enumerate(pair_labels)}

    return pair_label_dict


def get_captions(pair_label_dict, model, device, normalize=True):
    """
    Generates text embeddings for all attribute-object pair combinations.
    
    Args:
        pair_label_dict: Dictionary mapping pair labels to indices
        model: CLIP model for encoding text
        device: torch device (CPU/GPU)
        normalize: Whether to L2 normalize the embeddings
    
    Returns:
        captions: List of caption strings
        caption_embeddings: Tensor of encoded caption embeddings
    """
    # make captions out of all pair labels
    captions = []
    for pair_label in pair_label_dict.keys():
        caption = f"{pair_label[0]} {pair_label[1]} and {pair_label[2]} {pair_label[3]}"
        captions.append(caption) 

    # encode the captions
    with torch.no_grad():
        # make it batched
        caption_embeddings = []
        for i in range(0, len(captions), 512):
            caption_embeddings.append(model.encode_text(clip.tokenize(captions[i:i+512]).to(device)).float())

        caption_embeddings = torch.cat(caption_embeddings, dim=0)
        if normalize:
            caption_embeddings /= caption_embeddings.norm(dim=-1, keepdim=True)
        
    return captions, caption_embeddings



def get_caption_labels(pair_labels, pair_label_dict):
    """
    Converts pair labels to indices and generates negative pairs by swapping attributes.
    
    Args:
        pair_labels: List of attribute-object pairs
        pair_label_dict: Dictionary mapping pair labels to indices
    
    Returns:
        labels: Indices of positive pairs
        neg_labels: Indices of negative pairs (with swapped attributes)
    """
    labels = []
    neg_labels = []

    for pair_label in pair_labels:
        if tuple(pair_label) in pair_label_dict:
            labels.append(pair_label_dict[tuple(pair_label)])
        elif (pair_label[2], pair_label[3], pair_label[0], pair_label[1]) in pair_label_dict:
            labels.append(pair_label_dict[(pair_label[2], pair_label[3], pair_label[0], pair_label[1])])
        else:
            raise ValueError('The pair label is not in the pair label dictionary')

        neg_pair_label = (pair_label[2], pair_label[1], pair_label[0], pair_label[3])
        if tuple(neg_pair_label) in pair_label_dict:
            neg_labels.append(pair_label_dict[tuple(neg_pair_label)])
        elif (neg_pair_label[2], neg_pair_label[3], neg_pair_label[0], neg_pair_label[1]) in pair_label_dict:
            neg_labels.append(pair_label_dict[(neg_pair_label[2], neg_pair_label[3], neg_pair_label[0], neg_pair_label[1])])
        else:
            raise ValueError('The pair label is not in the pair label dictionary')

    return labels, neg_labels 

def get_image_embeddings(dataloader, model, device):
    """
    Generates CLIP image embeddings for all images in the dataloader.
    
    Args:
        dataloader: DataLoader containing images
        model: CLIP model for encoding images
        device: torch device (CPU/GPU)
    
    Returns:
        Normalized tensor of image embeddings
    """
    model.eval()

    image_embeddings = []
    with torch.no_grad():
        for images, _, _ in tqdm(dataloader):
            images = images.to(device)
            image_embedding = model.encode_image(images)
            image_embeddings.append(image_embedding)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

    return image_embeddings

def get_results_i2t(image_embeddings, caption_embeddings, labels, pair_label_dict, verbose=False):
    """
    Evaluates image-to-text retrieval performance.
    
    Args:
        image_embeddings: Tensor of encoded images
        caption_embeddings: Tensor of encoded captions
        labels: Ground truth label indices
        pair_label_dict: Dictionary mapping pair labels to indices
        verbose: Whether to print incorrect predictions
    
    Returns:
        Tuple of (top1_recall, top5_recall, top10_recall, reverse_recall)
    """
    # Calculate cosine similarity between all images and captions
    similarity = image_embeddings @ caption_embeddings.T

    pred_true_1 = 0  
    pred_true_5 = 0  
    pred_true_10 = 0  
    reverse_correct = 0  # Counter for predictions that match when attributes are reversed

    for i in range(len(labels)):    # for each image
        pred = similarity[i] 
        b = pred.argsort()    
        true_index = labels[i]   
        k = (b.flip(0) == true_index).nonzero().item()

        if true_index in b[-1:]:   # if the true label is in the top k predictions
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

            # check if the reversed attribute label is predicted
            if (pred[2], pred[1], pred[0], pred[3]) == true or (pred[0], pred[3], pred[2], pred[1]) == true:
                reverse_correct += 1
    
    print(f"Top 1 recall: {pred_true_1/len(similarity)}")
    print(f"Top 5 recall: {pred_true_5/len(similarity)}")
    print(f"Top 10 recall: {pred_true_10/len(similarity)}")
    print(f"Reverse recall: {reverse_correct/len(similarity)}")

    return pred_true_1/len(similarity), pred_true_5/len(similarity), pred_true_10/len(similarity), reverse_correct/len(similarity)


def get_accuracy(image_embeddings, caption_embeddings, pair_indices, neg_pair_indices):
    """
    Calculates accuracy by comparing similarity scores between positive and negative pairs.
    
    Args:
        image_embeddings: Tensor of encoded images
        caption_embeddings: Tensor of encoded captions
        pair_indices: Indices of positive pairs
        neg_pair_indices: Indices of negative pairs
    
    Returns:
        Accuracy score (correct predictions / total comparisons)
    """
    # get the caption embeddings for positve and negative pairs
    pos_caption_embeddings = caption_embeddings[pair_indices]
    neg_caption_embeddings = caption_embeddings[neg_pair_indices]

    # calculate cosine similarity between images and captions
    pos_similarity = torch.sum(image_embeddings * pos_caption_embeddings, dim=-1)
    neg_similarity = torch.sum(image_embeddings * neg_caption_embeddings, dim=-1)

    correct = 0
    total = 0
    for i in range(len(pos_similarity)):
        if pair_indices[i] != neg_pair_indices[i]:    # exclude pairs with the same attributes
            total += 1
            # compare similarity scores between positive and negative pairs
            if pos_similarity[i] > neg_similarity[i] or (pos_similarity[i] == neg_similarity[i]):
                correct += 1

    print(f"Accuracy: {correct/total}")

    return correct/total