import time
import torch
import torch.nn as nn
from thop import profile, clever_format

# ==========================================
# 1. Efficiency Metrics (Parameters, MACs, FLOPs)
# ==========================================

def calculate_efficiency_metrics(model, device="cuda"):
    """
    Calculates and prints the number of parameters, MACs, and FLOPs[cite: 38, 39].
    Requires the 'thop' library.
    """
    model.eval()
    # Dummy input based on standard ImageNet resolution (Batch Size 1, 3 Channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        
    # FLOPs are typically considered to be 2 * MACs in modern hardware calculations
    flops = 2 * macs 
    
    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
    flops_fmt = clever_format([flops], "%.3f")
    
    print("-" * 40)
    print(f"Model Efficiency Metrics:")
    print(f"Total Parameters: {params_fmt}")
    print(f"MACs: {macs_fmt}")
    print(f"FLOPs: {flops_fmt}")
    print("-" * 40)
    
    return params, macs, flops

# ==========================================
# 2. Gradient Norm Tracking (Scenario 4.2)
# ==========================================

def compute_gradient_norms(model):
    """
    Extracts the L2 norm of the gradients for each parameter layer.
    Used for layer-wise adaptation analysis in fine-tuning.
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Calculate L2 norm of the gradient
            norm = param.grad.data.norm(2).item()
            grad_norms[name] = norm
    return grad_norms

# ==========================================
# 3. Training & Evaluation Loops
# ==========================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, track_grads=False):
    """
    Standard training loop for a single epoch.
    Optionally tracks gradient norms for Scenario 4.2.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_grad_norms = {}
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Track gradient norms before the optimizer step if requested
        if track_grads:
            batch_grads = compute_gradient_norms(model)
            for name, val in batch_grads.items():
                epoch_grad_norms[name] = epoch_grad_norms.get(name, 0.0) + val
                
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Average the gradient norms over the batches
    if track_grads:
        for name in epoch_grad_norms:
            epoch_grad_norms[name] /= len(dataloader)
            
    return epoch_loss, epoch_acc, epoch_grad_norms


def evaluate(model, dataloader, criterion, device):
    """
    Standard evaluation loop. 
    Metrics should be printed during evaluation.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Store all predictions and targets (useful for confusion matrix later)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, track_grads=False):
    """
    Full training pipeline enforcing the maximum computational constraints.
    Max 30 epochs for full-data [cite: 120], Max 20 epochs for few-shot[cite: 121].
    Max 6 hours per model per scenario[cite: 122].
    """
    history = {
        "train_loss": [], "train_acc": [], 
        "val_loss": [], "val_acc": [],
        "grad_norms": []
    }
    
    start_time = time.time()
    MAX_TIME_SECONDS = 6 * 60 * 60 # 6 hours [cite: 122]
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc, grad_norms = train_one_epoch(
            model, train_loader, criterion, optimizer, device, track_grads
        )
        
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if track_grads:
            history["grad_norms"].append(grad_norms)
            
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Time: {time.time() - epoch_start_time:.2f}s")
              
        # Enforce the 6-hour time limit constraint [cite: 122]
        if time.time() - start_time > MAX_TIME_SECONDS:
            print("WARNING: Reached the 6-hour computational limit. Stopping training early.")
            break
            
    return history