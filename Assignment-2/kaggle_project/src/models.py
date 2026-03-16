import torch
import torch.nn as nn
import timm

# ==========================================
# 1. Base Model Loading
# ==========================================

def get_model(model_name, num_classes=30, pretrained=True):
    """
    Loads a pre-trained model from timm and modifies the classification head.
    Allowed models: 'resnet50', 'inception_v3', 'densenet121', 
                    'efficientnet_b0', 'convnext_tiny'
    """
    # Create model with pre-trained weights  and replace the classifier head for 30 classes 
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

# ==========================================
# 2. Freezing & Fine-Tuning Logic (Scenarios 4.1 & 4.2)
# ==========================================

def setup_finetuning_strategy(model, model_name, strategy="linear_probe", unfreeze_percent=0.20):
    """
    Applies the specific freezing/unfreezing strategy required for Scenarios 4.1 and 4.2.
    
    Strategies:
    - "linear_probe": Freeze all except the final classification layer [cite: 42-46, 55].
    - "last_block": Freeze all except the last convolutional block and the head[cite: 56].
    - "full": Unfreeze all parameters[cite: 57].
    - "selective": Unfreeze exactly 'unfreeze_percent' (e.g., 20%) of the parameters from the top down.
    """
    
    # Strategy 1: Full Fine-Tuning [cite: 57]
    if strategy == "full":
        for param in model.parameters():
            param.requires_grad = True
        return model

    # Strategy 2: Linear Probe [cite: 42-46, 55]
    if strategy == "linear_probe":
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze the classification head. timm models uniformly use 'get_classifier()' 
        # or we can just check the parameter names.
        for name, param in model.named_parameters():
            if 'head' in name or 'fc' in name or 'classifier' in name:
                param.requires_grad = True
        return model

    # Strategy 3: Last Block Fine-Tuning [cite: 56]
    if strategy == "last_block":
        for param in model.parameters():
            param.requires_grad = False
            
        for name, param in model.named_parameters():
            # Always unfreeze the head
            if 'head' in name or 'fc' in name or 'classifier' in name:
                param.requires_grad = True
            
            # Unfreeze the specific last block based on the architecture
            if model_name == "resnet50" and "layer4" in name:
                param.requires_grad = True
            elif model_name == "efficientnet_b0" and "blocks.6" in name:
                param.requires_grad = True
            elif model_name == "convnext_tiny" and "stages.3" in name:
                param.requires_grad = True
                
        return model

    # Strategy 4: Selective Unfreezing (Max 20%) 
    if strategy == "selective":
        total_params = sum(p.numel() for p in model.parameters())
        target_unfrozen = total_params * unfreeze_percent
        
        # Start by freezing everything
        for param in model.parameters():
            param.requires_grad = False
            
        # Iterate backwards (from the classifier down into the deep layers)
        current_unfrozen = 0
        for name, param in reversed(list(model.named_parameters())):
            if current_unfrozen + param.numel() <= target_unfrozen:
                param.requires_grad = True
                current_unfrozen += param.numel()
            else:
                # We've hit our 20% limit 
                break
                
        return model

    raise ValueError(f"Unknown fine-tuning strategy: {strategy}")

# ==========================================
# 3. Layer-Wise Feature Extraction (Scenario 4.5)
# ==========================================

class FeatureExtractor(nn.Module):
    """
    Extracts intermediate features from early, middle, and final layers .
    timm makes this highly convenient using 'features_only=True'.
    """
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        
        # features_only=True returns a list of feature maps from different depths[cite: 100].
        # By default, timm usually returns 4 or 5 feature maps corresponding to the network stages.
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True
        )
        
        # Freeze the backbone completely as we are just probing the features [cite: 101]
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Global Average Pooling to flatten spatial dimensions into feature vectors
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # features is a list of tensors from different depths
        features = self.backbone(x) 
        
        pooled_features = []
        for f in features:
            # Pool to (B, C, 1, 1) and flatten to (B, C)
            pooled = self.pool(f).flatten(1)
            pooled_features.append(pooled)
            
        # We return a dictionary explicitly mapping to early, middle, and final [cite: 100]
        # (Assuming the model returns 4 feature maps, which is standard for ResNet/ConvNeXt)
        if len(pooled_features) >= 3:
            return {
                "early": pooled_features[0],
                "middle": pooled_features[len(pooled_features) // 2],
                "final": pooled_features[-1]
            }
        else:
            return {"features": pooled_features}