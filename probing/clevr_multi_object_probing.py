import torch
import clip
from torch.utils.data import DataLoader
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loading.clevr import *
from probing_datasets import *
from probing_models import *

def train_val_test_split_multi(all_pair_labels, split_ratio=0.1, seed=42):
    """
    Split the dataset into train, validation and test sets while ensuring unique object-attribute combinations
    are properly distributed across splits.
    
    Args:
        all_pair_labels: List of object-attribute pair labels
        split_ratio: Proportion of data to use for test and validation sets
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    unique_pair_labels = get_unique_combinations(all_pair_labels)

    # sample combinations for testing and validation
    sampled_matrices_test = [] 
    sampled_matrices_val = []

    while len(sampled_matrices_test) < int(split_ratio * len(unique_pair_labels)):
        sample = random.sample(unique_pair_labels, 1)[0]
        matrix = tuple_to_matrix(sample)
        if any(np.array_equal(matrix, m) for m in sampled_matrices_test):
            continue 
        sampled_matrices_test.append(matrix)

    while len(sampled_matrices_val) < int(split_ratio * len(unique_pair_labels)):
        sample = random.sample(unique_pair_labels, 1)[0]
        matrix = tuple_to_matrix(sample)
        if any(np.array_equal(matrix, m) for m in sampled_matrices_val) or any(np.array_equal(matrix, m) for m in sampled_matrices_test):
            continue
        sampled_matrices_val.append(matrix)

    # find the indices which are in the sampled_pairs, that is the test and validation sets
    test_indices = []
    val_indices = []
    train_indices = []

    for i, pair_label in enumerate(all_pair_labels):
        matrix = tuple_to_matrix(pair_label)
        if any(np.array_equal(matrix, m) for m in sampled_matrices_test):
            test_indices.append(i)
        elif any(np.array_equal(matrix, m) for m in sampled_matrices_val):
            val_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, val_indices, test_indices


def process_labels_multi(pair_labels, target_obj):
    """
    Process labels for multi-object probing by creating probability distributions
    for attributes of the target object.
    
    Args:
        pair_labels: List of object-attribute pairs
        target_obj: The object type to probe for (e.g., 'cube', 'sphere')
    Returns:
        Tensor of probability distributions for each attribute
    """
    labels = []
    indices = []
    labels_list = []

    for i, pair_label in enumerate(pair_labels):
        if target_obj in pair_label:    # pair labels 
            # find all positions of the target object in the pair label
            target_obj_indices = [i-1 for i, obj in enumerate(pair_label) if obj == target_obj]
            target_attr_labels = [attr_to_idx[pair_label[i]] for i in target_obj_indices]
            # create a probability distribution for the target attribute based on the number of times it appears
            target_attr_probs = [target_attr_labels.count(attr) / len(target_attr_labels) for attr in attr_to_idx.values()]
            labels.append(target_attr_probs)
            # indices.append(i)
            # labels_list.append(target_attr_labels)

    labels = torch.tensor(labels)
    return labels


def calculate_accuracy_multi(model, device, dataloader):
    """
    Calculate accuracy for multi-object probing, considering multiple correct answers.
    Accuracy is 1 if the model predicts all correct attributes for an object.
    
    Args:
        model: The probing model
        device: Device to run computations on
        dataloader: DataLoader containing the evaluation data
    Returns:
        Float representing accuracy
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels_batched = labels.to(device)

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)

            for i, prob in enumerate(probabilities):
                labels = labels_batched[i].nonzero().squeeze(1).tolist()
                top_probs, top_indices = torch.topk(prob, len(labels))

                top_indices = set(top_indices.tolist())
                # print(top_indices, labels)

                if set(top_indices) == set(labels):
                    correct += 1
                total += 1

    return correct / total

def train_model_multi(model, device, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    """
    Train the probing model using the provided data loaders.
    Prints training progress every 10% of total epochs.
    
    Args:
        model: The probing model to train
        device: Device to run computations on
        criterion: Loss function
        optimizer: Optimizer for training
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
    """
    for epoch in range(num_epochs):

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # _, predicted = torch.max(outputs, 1)
            # train_correct += (predicted == labels).sum().item()
            # train_total += labels.size(0)

        train_loss /= len(train_loader)
        # train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # _, predicted = torch.max(outputs, 1)
                # val_correct += (predicted == labels).sum().item()
                # val_total += labels.size(0)

        val_loss /= len(val_loader)
        # val_accuracy = val_correct / val_total

        if epoch % (num_epochs / 10) == 0:
            # print(f'Epoch: {epoch}. Train Loss: {train_loss:.4f}. Train Accuracy: {train_accuracy:.4f}. Val Loss: {val_loss:.4f}. Val Accuracy: {val_accuracy:.4f}')
            print(f'Epoch: {epoch}. Train Loss: {train_loss:.4f}. Val Loss: {val_loss:.4f}')

def probing_multi(target_obj, all_embeddings, all_pair_labels, batch_size, num_epochs, lr, weight_decay=0, device='cuda'):
    """
    Main probing function for multi-object scenes. Sets up and runs the probing experiment
    for a specific target object.
    """
    # get the embeddings and pair labels for the target object
    target_indices = target_obj_split(target_obj, all_pair_labels)
    print(f"\nThe number of samples for the object {target_obj}: {len(target_indices)}")
    pair_labels = [all_pair_labels[i] for i in target_indices]
    embeddings = all_embeddings[target_indices]

    # split the data into train, validation and test sets
    train_indices, val_indices, test_indices = train_val_test_split_multi(pair_labels)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}\n")

    train_embeddings = embeddings[train_indices]
    val_embeddings = embeddings[val_indices]
    test_embeddings = embeddings[test_indices]

    train_pair_labels = [pair_labels[i] for i in train_indices]
    val_pair_labels = [pair_labels[i] for i in val_indices]
    test_pair_labels = [pair_labels[i] for i in test_indices]

    # get the adjective labels for the target object
    train_labels = process_labels_multi(train_pair_labels, target_obj)
    val_labels = process_labels_multi(val_pair_labels, target_obj)
    test_labels = process_labels_multi(test_pair_labels, target_obj)

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
    
    train_model_multi(model, device, criterion, optimizer, train_loader, val_loader, num_epochs)

    train_accuracy = calculate_accuracy_multi(model, device, train_loader)
    val_accuracy = calculate_accuracy_multi(model, device, val_loader)
    test_accuracy = calculate_accuracy_multi(model, device, test_loader)

    print(f"Training accuracy: {train_accuracy}")
    print(f"Validation accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    return train_accuracy, val_accuracy, test_accuracy

def load_embeddings(embedding_path):
    if os.path.exists(embedding_path):
        return torch.load(embedding_path, map_location=device).squeeze(1)
    return None

def save_embeddings(embedding_path, embeddings):
    torch.save(embeddings, embedding_path)

def main(args):
    # Load dataset
    clevr_loader = CLEVRLoader(args.data_path, num_objects=args.num_objects, download=args.download)
    all_filenames, all_pair_labels = clevr_loader.filenames, clevr_loader.pair_labels

    model, preprocess = clip.load(args.clip_model, device=device)

    if args.probe_type == "image":
        image_embeddings = load_embeddings(args.embedding_path)
        
        if image_embeddings is None:
            image_embeddings = encode_images(all_filenames, model, preprocess, device).squeeze(1)
            save_embeddings(args.embedding_path, image_embeddings)
        
        avg_train_acc, avg_val_acc, avg_test_acc = 0, 0, 0
        for target_obj in ['cube', 'sphere', 'cylinder']:
            train_acc, val_acc, test_acc = probing_multi(
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
        tokenized_captions = preprocess_captions(all_pair_labels).to(device)
        text_embeddings = get_text_embeddings(tokenized_captions, model, device).float()
        
        avg_train_acc, avg_val_acc, avg_test_acc = 0, 0, 0
        for target_obj in ['cube', 'sphere', 'cylinder']:
            train_acc, val_acc, test_acc = probing_multi(
                target_obj, text_embeddings, all_pair_labels, args.batch_size, args.epochs, args.lr, device=device
            )
            avg_train_acc += train_acc
            avg_val_acc += val_acc
            avg_test_acc += test_acc
        avg_train_acc /= 3
        avg_val_acc /= 3
        avg_test_acc /= 3
        print(f"Average: Train {avg_train_acc:.4f}, Val {avg_val_acc:.4f}, Test {avg_test_acc:.4f}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to the CLEVR dataset")
    parser.add_argument("--probe_type", type=str, choices=["image", "text"], required=True,
                      help="Whether to probe image or text embeddings")
    parser.add_argument("--num_objects", type=int, default=2,
                      help="Number of objects in the scene")
    parser.add_argument("--embedding_path", type=str, default="../cache/clevr_multi_img_emb.pt",
                      help="Path to save/load pre-computed embeddings")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14",
                      help="CLIP model variant to use")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-1,
                      help="Learning rate")
    parser.add_argument("--download", action="store_true",
                      help="Download dataset if not present")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)

