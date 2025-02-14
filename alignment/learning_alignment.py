import torch
from tqdm import tqdm
import wandb

# Class for learning a linear alignment transformation for CLIP embeddings
class CLIPAlignment(torch.nn.Module): 
    def __init__(self, dim):
        super(CLIPAlignment, self).__init__()
        # Initialize linear transformation as identity matrix
        self.linear = torch.nn.Linear(dim, dim, bias=False, dtype=torch.float32)
        self.linear.weight.data = torch.eye(dim, dtype=torch.float32)
        # Learnable temperature parameter for scaling logits
        self.t = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x): 
        # Apply linear transformation and normalize
        x = self.linear(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

# Class for fine-tuning CLIP model with alignment
class CLIPFTAlignment(torch.nn.Module):    
    def __init__(self, model, dim):
        super(CLIPFTAlignment, self).__init__()
        self.model = model
        # Initialize alignment layer similar to CLIPAlignment but with float16
        self.linear = torch.nn.Linear(dim, dim, bias=False, dtype=torch.float16)
        self.linear.weight.data = torch.eye(dim, dtype=torch.float16)
        self.t = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float16))

    def forward(self, image, caption, neg_caption):
        # Encode inputs using CLIP
        x = self.model.encode_image(image)
        y = self.model.encode_text(caption)
        z = self.model.encode_text(neg_caption)
        # Apply alignment to text embeddings
        y = self.linear(y)
        z = self.linear(z)
        # Normalize all embeddings
        x = x / x.norm(dim=-1, keepdim=True)
        y = y / y.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)
        # Combine positive and negative text embeddings
        y = torch.cat([y, z], dim=0)

        return x, y
    
    def encode_text(self, text):
        # Helper method for encoding and aligning text
        x = self.model.encode_text(text)
        x = self.linear(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def encode_image(self, image):
        # Helper method for encoding images
        x = self.model.encode_image(image)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    

def constrastive_loss(image_embeddings, new_text_embeddings, t, device):
    '''
    Computes symmetric contrastive loss between image and text embeddings
    
    Args:
        image_embeddings: [batch_size, d] normalized image embeddings
        new_text_embeddings: [batch_size, d] normalized text embeddings
        t: temperature parameter
        device: torch device
    '''
    # Compute similarity matrix
    logits =  image_embeddings @ new_text_embeddings.T # batch_size x batch_size
    labels = torch.arange(logits.shape[0]).to(device)  # [batch_size]

    # Compute bidirectional cross entropy loss
    loss1 = torch.nn.functional.cross_entropy(torch.exp(t) * logits, labels)
    loss2 = torch.nn.functional.cross_entropy(torch.exp(t) * logits.T, labels)
    loss = (loss1 + loss2) / 2

    # Compute accuracy
    argmax_preds = torch.argmax(logits, dim=1)
    accuracy = (argmax_preds == labels).sum().item() / labels.shape[0]

    return loss, accuracy


def constrastive_loss_with_negatives(image_embeddings, new_text_embeddings, t, device):
    '''
    Computes contrastive loss with negative examples
    
    Args:
        image_embeddings: [batch_size, d] normalized image embeddings
        new_text_embeddings: [2*batch_size, d] normalized text embeddings (positives + negatives)
        t: temperature parameter
        device: torch device
    '''
    logits =  image_embeddings @ new_text_embeddings.T    # batch_size x (2*batch_size)

    labels = torch.arange(logits.shape[0]).to(device)    # [batch_size]
    one_hot_labels = torch.eye(n=logits.shape[0], m=logits.shape[1], device=device)    # [batch_size, 2*batch_size]

    # Compute image-to-text and text-to-image losses
    loss1 = torch.nn.functional.cross_entropy(torch.exp(t) * logits, one_hot_labels)    # I2T
    loss2 = torch.nn.functional.cross_entropy(torch.exp(t) * logits.T[:logits.shape[0],:], labels)    # T2I
    loss = (loss1 + loss2) / 2

    # Compute accuracy
    argmax_preds = torch.argmax(logits, dim=1)
    accuracy = (argmax_preds == labels).sum().item() / labels.shape[0]

    return loss, accuracy

# Training function for model with negative examples
def train_model_neg(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10, scheduler=None):

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        # Training
        for data in train_loader:
            image_embeddings, text_embeddings, neg_text_embeddings = data   
            image_embeddings = image_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)
            neg_text_embeddings = neg_text_embeddings.to(device)     
            
            optimizer.zero_grad()

            text_embeddings = model(text_embeddings)    # [d, d] @ [d, batch_size] = [d, batch_size]
            neg_text_embeddings = model(neg_text_embeddings)    # [d, d] @ [d, batch_size] = [d, batch_size]
            new_text_embeddings = torch.cat([text_embeddings, neg_text_embeddings], dim=0) # [batch_size+neg_samples, d]

            loss, accuracy_report = loss_fn(image_embeddings.squeeze(1), new_text_embeddings.squeeze(1), model.t, device)
            train_loss += loss
            train_accuracy += accuracy_report

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        train_accuracy = train_accuracy / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad(): 
            for data in val_loader:
                image_embeddings, text_embeddings, neg_text_embeddings = data
                image_embeddings = image_embeddings.to(device)
                text_embeddings = text_embeddings.to(device)
                neg_text_embeddings = neg_text_embeddings.to(device)
                
                text_embeddings = model(text_embeddings)    # [d, d] @ [d, batch_size] = [d, batch_size]
                neg_text_embeddings = model(neg_text_embeddings)    # [d, d] @ [d, batch_size] = [d, batch_size]
                new_text_embeddings = torch.cat([text_embeddings, neg_text_embeddings], dim=0)
                
                loss, accuracy_report = loss_fn(image_embeddings.squeeze(1), new_text_embeddings.squeeze(1), model.t, device)

                val_loss += loss
                val_accuracy += accuracy_report
                
        if scheduler:
            scheduler.step()

        val_loss = val_loss / len(val_loader)
        val_accuracy = val_accuracy / len(val_loader)

        # print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}')
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

# Training function for fine-tuning with negative examples
def train_ft_model_neg(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10, scheduler=None):

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        # Training
        for data in train_loader:
            images, captions, neg_captions = data   
            images = images.to(device)
            captions = captions.to(device)
            neg_captions = neg_captions.to(device)
            
            optimizer.zero_grad()

            image_embeddings, text_embeddings = model(images, captions, neg_captions)

            loss, accuracy_report = loss_fn(image_embeddings.squeeze(1), text_embeddings.squeeze(1), model.t, device)
            train_loss += loss
            train_accuracy += accuracy_report

            loss.backward()
            optimizer.step() 

        train_loss = train_loss / len(train_loader)
        train_accuracy = train_accuracy / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad(): 
            for data in val_loader:
                images, captions, neg_captions = data
                images = images.to(device)
                captions = captions.to(device)
                neg_captions = neg_captions.to(device)
                
                image_embeddings, text_embeddings = model(images, captions, neg_captions)
                
                # print(image_embeddings.shape, text_embeddings.shape)
                loss, accuracy_report = loss_fn(image_embeddings.squeeze(1), text_embeddings.squeeze(1), model.t, device)

                val_loss += loss
                val_accuracy += accuracy_report
                
        if scheduler:
            scheduler.step()

        val_loss = val_loss / len(val_loader)
        val_accuracy = val_accuracy / len(val_loader)

        # print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}')
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})


def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10, scheduler=None):

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        # Training
        for data in train_loader:
            image_embeddings, text_embeddings = data   
            image_embeddings = image_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)
            
            optimizer.zero_grad()

            text_embeddings = model(text_embeddings)    # [d, d] @ [d, batch_size] = [d, batch_size]
            loss, accuracy_report = loss_fn(image_embeddings.squeeze(1), text_embeddings.squeeze(1), model.t, device)
            train_loss += loss
            train_accuracy += accuracy_report

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        train_accuracy = train_accuracy / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad(): 
            for data in val_loader:
                image_embeddings, text_embeddings = data
                image_embeddings = image_embeddings.to(device)
                text_embeddings = text_embeddings.to(device)
                
                text_embeddings = model(text_embeddings)    # [d, d] @ [d, batch_size] = [d, batch_size]
                
                loss, accuracy_report = loss_fn(image_embeddings.squeeze(1), text_embeddings.squeeze(1), model.t, device)

                val_loss += loss
                val_accuracy += accuracy_report
                
        if scheduler:
            scheduler.step()

        val_loss = val_loss / len(val_loader)
        val_accuracy = val_accuracy / len(val_loader)

        # print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}')
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})
