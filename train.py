from torch import nn
import torch
from dataset import build_data
import os 
from transformer import Decoder, create_causal_mask
from torch.optim.lr_scheduler import StepLR


device = 'cuda'

train_file_path = os.path.join("data", "train.csv")
char_to_idx, idx_to_char, vocab_size, train_data = build_data(train_file_path)
train_data = torch.Tensor(train_data).to(dtype=torch.long)  # nn.Embedding expects type Long or Int

n_epochs = 200
batch_size = 64
seq_len = 150
lr = 1e-3

n_sequences = train_data.size(0) // seq_len
train_data = train_data[:n_sequences * seq_len].view(-1, seq_len)

d_model = 128
d_hidden = 256
n_heads = 4
n_layers = 4

cross_entropy_loss = nn.CrossEntropyLoss()
model = Decoder(vocab_size, d_model, d_hidden, n_heads, n_layers).to(device)
model.train()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = StepLR(optimizer, step_size=5, gamma=0.95)

for epoch in range(n_epochs):
    for j in range(0, n_sequences, batch_size):
        x = train_data[j:j + batch_size, :].to(device)

        inputs = x[:, :-1]
        targets = x[:, 1:]
        causal_mask = create_causal_mask(seq_len - 1).to(device)  # mask size is equal to truncated seq_len due to shifting 

        predictions = model(inputs, causal_mask)
        
        predictions = predictions.view(-1, vocab_size)  # reshape to (bsz * seq_len, vocab_size)
        targets = targets.flatten()
        loss = cross_entropy_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if j % 100 == 0:
            print(f"Epoch {epoch}, Step {j}, Loss: {loss.item():.4f}")

    
torch.save(model.state_dict(), 'model.pth')