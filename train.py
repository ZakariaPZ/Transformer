from torch import nn
import torch
from dataset import build_data
import os 
from transformer import Decoder, create_causal_mask
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


# Load dataset
train_file_path = os.path.join("data", "train.csv")
val_file_path = os.path.join("data", "validation.csv")
char_to_idx, idx_to_char, vocab_size, train_data, val_data = build_data(train_file_path, val_file_path)
train_data = torch.Tensor(train_data).to(dtype=torch.long)
val_data = torch.tensor(val_data, dtype=torch.long)

# Hyperparameters
n_epochs = 250
batch_size = 64
seq_len = 64
lr = 1e-3
dropout = 0.2

n_sequences = train_data.size(0) // seq_len
train_data = train_data[:n_sequences * seq_len].view(n_sequences, seq_len)
n_val_sequences = val_data.size(0) // seq_len
val_data = val_data[:n_val_sequences * seq_len].view(n_val_sequences, seq_len)

def get_batch(batch_size):
    batch_idxs = torch.randint(n_sequences, size=(batch_size,))
    return train_data[batch_idxs]

# Architecture parameters 
d_model = 128
d_head = 128
d_hidden = 512
n_heads = 6
n_layers = 6

device = 'cuda'
model = Decoder(vocab_size, d_model, d_head, d_hidden, n_heads, n_layers, dropout=dropout).to(device)
# Print number of parameters in model
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Total number of parameters in model (M): {total_params:,}")

cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

train_losses = []
val_losses = []

for epoch in range(n_epochs):

    model.train()
    total_train_loss = 0
    n_train_batches = 0

    for j in range(0, n_sequences, batch_size):
        x = get_batch(batch_size).to(device)

        inputs = x[:, :-1]
        targets = x[:, 1:]
        causal_mask = create_causal_mask(seq_len - 1).to(device)  # mask size is equal to truncated sequence length due to shifting 

        predictions = model(inputs, causal_mask)
        
        predictions = predictions.view(-1, vocab_size)  # reshape to (B * N, V)
        targets = targets.flatten()
        loss = cross_entropy_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        n_train_batches += 1

    avg_train_loss = total_train_loss / n_train_batches
    print(f"Epoch {epoch} Training Loss: {avg_train_loss:.4f}")
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    n_val_batches = 0
    
    with torch.no_grad():
        for j in range(0, n_val_sequences, batch_size):
            # Get validation batch
            val_batch = val_data[j:j + batch_size].to(device)

            # Create inputs and targets
            val_inputs = val_batch[:, :-1]
            val_targets = val_batch[:, 1:]
            val_mask = create_causal_mask(seq_len - 1).to(device)
            
            # Get predictions
            val_predictions = model(val_inputs, val_mask)
            
            # Calculate loss
            val_predictions = val_predictions.view(-1, vocab_size)
            val_targets = val_targets.flatten()
            val_loss = cross_entropy_loss(val_predictions, val_targets)
            
            total_val_loss += val_loss.item()
            n_val_batches += 1

    avg_val_loss = total_val_loss / n_val_batches
    print(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}")
    val_losses.append(avg_val_loss)

    if epoch % 50 == 0 or epoch == n_epochs - 1:
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        torch.save(checkpoint, os.path.join('checkpoints', f'checkpoint_epoch_{epoch}.pt'))
        print(f"Checkpoint saved at epoch {epoch}")

    scheduler.step(avg_val_loss)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join('plots', 'loss_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
