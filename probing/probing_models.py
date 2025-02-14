import torch

class CLIPProbing(torch.nn.Module):
    """
    Basic linear probing model that maps CLIP embeddings to object attribute predictions.
    Used for analyzing how well CLIP's frozen representations capture attribute-object binding.
    """
    def __init__(self, n_inputs, n_outputs):
        super(CLIPProbing, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs, dtype=torch.float32)

    def forward(self, x):
        x = self.linear(x)  
        return x
    

class CLIPFTProbingImage(torch.nn.Module):    
    """
    Probing model that fine-tunes CLIP's image encoder for object attribute prediction.
    Unlike CLIPProbing, this model updates CLIP's image encoder weights during training.
    """
    def __init__(self, model, n_inputs, n_outputs):
        super(CLIPFTProbingImage, self).__init__()
        self.model = model  # CLIP model for image encoding
        # Linear layer for mapping CLIP image embeddings to task outputs
        self.linear = torch.nn.Linear(n_inputs, n_outputs, dtype=torch.float16)

    def forward(self, image):
        # Get image embeddings from CLIP
        x = self.model.encode_image(image)
        # Project embeddings to output space
        x = self.linear(x)  
        return x
    

class CLIPFTProbingText(torch.nn.Module):    
    """
    Probing model that fine-tunes CLIP's text encoder for object attribute prediction.    
    """
    def __init__(self, model, n_inputs, n_outputs):
        super(CLIPFTProbingText, self).__init__()
        self.model = model  # CLIP model for text encoding
        # Linear layer for mapping CLIP text embeddings to task outputs
        self.linear = torch.nn.Linear(n_inputs, n_outputs, dtype=torch.float16)

    def forward(self, text):
        # Get text embeddings from CLIP
        x = self.model.encode_text(text)
        # Project embeddings to output space
        x = self.linear(x)  
        return x
    

def calculate_accuracy(model, device, dataloader):
    """
    Calculate the accuracy of attribute predictions on the given dataset.
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def train_model(model, device, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    """
    Train a probing model to predict object attributes and evaluate on validation set.
    
    Args:
        model: The probing model to train
        device: Device (CPU/GPU) to run training on
        criterion: Loss function for attribute prediction
        optimizer: Optimizer for updating model parameters
        train_loader: DataLoader containing training samples and attribute labels
        val_loader: DataLoader containing validation samples and attribute labels
        num_epochs: Number of training epochs
        
    The function prints training progress including loss and accuracy metrics
    for both training and validation sets at regular intervals.
    """
    
    for epoch in range(num_epochs):

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass and optimization
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track training metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Track validation metrics
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        # Print progress every 10% of epochs
        if epoch % (num_epochs / 10) == 0:
            print(f'Epoch: {epoch}. Train Loss: {train_loss:.4f}. Train Accuracy: {train_accuracy:.4f}. Val Loss: {val_loss:.4f}. Val Accuracy: {val_accuracy:.4f}')
