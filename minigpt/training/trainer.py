import torch
from torch.utils.data import DataLoader

def train(model, dataset, config):
    """Train the model on the dataset"""
    model.train()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        total_loss = 0.0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(config.device), y.to(config.device)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()
            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step}: loss = {loss.item():.4f}")
        print(f"Epoch {epoch} completed with average loss: {total_loss / len(dataloader):.4f}")
    
    return model

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.9, do_sample=True, top_k=None):
    """Generate text using the trained model"""
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(next(model.parameters()).device)
    
    with torch.no_grad():
        output = model.generate(
            input_tensor, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample, 
            temperature=temperature,
            top_k=top_k
        )
        generated = tokenizer.decode(output[0].tolist())
    
    return generated
