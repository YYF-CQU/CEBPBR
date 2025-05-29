import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define the BCELoss

loss_fn = nn.BCELoss()


# Specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Model Definition
def get_model():
    return CEBPBR()


model = get_model()


try:
    model = model.to(device)
    print(f"Model successfully moved to {device}.")
except RuntimeError as e:
    print(f"Error moving model to {device}: {e}")
    raise


for name, param in model.named_parameters():
    print(f"Parameter '{name}' is on device: {param.device}")


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("Optimizer successfully initialized.")


# Input data
batch_size = 4
height, width = 192, 192
data = torch.randn((batch_size, 3, height, width)).to(device)  # [B, 3, 192, 192]

# Segmentation target
# Shape: [B, 1, 192, 192]
target = torch.randint(0, 2, (batch_size, 1, height, width)).float().to(device)

# Boundary target full: [B, 1, 192, 192]
boundary_target_full = torch.randint(0, 2, (batch_size, 1, height, width)).float().to(device)

# Boundary target low-res: [B, 1, 6, 6]
boundary_height, boundary_width = 6, 6  
boundary_target_low_res = F.interpolate(
    boundary_target_full, 
    size=(boundary_height, boundary_width), 
    mode='bilinear', 
    align_corners=False
) 


# Set model to training mode
model.train()

try:
    segmentation_output, boundary_output, aux_boundary_outputs = model(data)  
    print("Model forward pass successful on CPU.")
except RuntimeError as e:
    print("Error during model forward pass on CPU:", e)
    raise


# Upsampling (If Necessary)
segmentation_upsampled = segmentation_output  # [B, 1, 192, 192]



print(f"Segmentation Output min: {segmentation_output.min().item()}, max: {segmentation_output.max().item()}") 
print(f"Boundary Output min: {boundary_output.min().item()}, max: {boundary_output.max().item()}")          

# Apply sigmoid to boundary_output
boundary_sigmoid = torch.sigmoid(boundary_output)
print(f"Boundary Sigmoid Output min: {boundary_sigmoid.min().item()}, max: {boundary_sigmoid.max().item()}")  


# Verify that the batch sizes match
assert segmentation_upsampled.shape[0] == target.shape[0], \
    f"Segmentation output batch size ({segmentation_upsampled.shape[0]}) does not match target batch size ({target.shape[0]})"
assert boundary_output.shape[0] == boundary_target_low_res.shape[0], \
    f"Boundary output batch size ({boundary_output.shape[0]}) does not match boundary target batch size ({boundary_target_low_res.shape[0]})"
for idx, aux in enumerate(aux_boundary_outputs):
    assert aux.shape[0] == boundary_target_full.shape[0], \
        f"Auxiliary boundary output {idx+1} batch size ({aux.shape[0]}) does not match boundary target batch size ({boundary_target_full.shape[0]})"


# Loss Computation
loss_segmentation = loss_fn(segmentation_upsampled, target)

# Calculate boundary loss using BCELoss
loss_boundary = loss_fn(boundary_sigmoid, boundary_target_low_res)

# Calculate auxiliary boundary losses using BCELoss
loss_aux_boundary = 0
for idx, aux_boundary in enumerate(aux_boundary_outputs):
    loss_aux = loss_fn(aux_boundary, boundary_target_full)
    loss_aux_boundary += loss_aux
    print(f"Auxiliary Boundary Loss {idx+1}: {loss_aux.item()}")

# Combine the segmentation loss and boundary loss with weighting factors
alpha = 1.0  # Weight for segmentation loss
beta = 0.5   # Weight for boundary loss
gamma = 0.3  # Weight for auxiliary boundary losses
loss = alpha * loss_segmentation + beta * loss_boundary + gamma * loss_aux_boundary

print(f"Segmentation Loss: {loss_segmentation.item():.4f}")
print(f"Boundary Loss: {loss_boundary.item():.4f}")
print(f"Auxiliary Boundary Loss: {loss_aux_boundary.item():.4f}")
print(f"Total Loss: {loss.item():.4f}")


print("Verifying optimizer's state tensor devices:")
for state_idx, state in enumerate(optimizer.state.values()):
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            print(f"Optimizer State Tensor '{key}' for state {state_idx} is on device: {value.device}")

optimizer_state_on_cuda = False
for state in optimizer.state.values():
    for value in state.values():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            optimizer_state_on_cuda = True
            break

if optimizer_state_on_cuda:
    print("Optimizer state tensors are on CUDA. Reinitializing optimizer to move state to CPU.")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Optimizer reinitialized successfully.")


optimizer.zero_grad()


loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


try:
    optimizer.step()
    print("Optimizer step completed successfully.")
except RuntimeError as e:
    print("Error during optimizer step:", e)
    raise




# Below is the main loss function (tversky_bce_loss)

def tversky_bce_loss(
    predictions, 
    ground_truths, 
    boundary_predictions=None, 
    boundary_ground_truths=None, 
    smooth=1e-8, 
    alpha=0.5, 
    beta=0.5, 
    pos_weight=1.0, 
    boundary_weight=0.5
):
    """
    Combined Tversky Loss and Weighted Binary Cross-Entropy Loss for binary segmentation and boundary refinement with deep supervision.
    """
    # Segmentation Loss
    # Flatten the predictions and ground truths for segmentation loss
    predictions_flat = predictions.view(-1)
    ground_truths_flat = ground_truths.view(-1).float()
    predictions_sigmoid = torch.sigmoid(predictions_flat)

    # Calculate Tversky loss
    TP = (predictions_sigmoid * ground_truths_flat).sum()
    FP = ((1 - ground_truths_flat) * predictions_sigmoid).sum()
    FN = (ground_truths_flat * (1 - predictions_sigmoid)).sum()

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    tversky_loss = 1 - tversky  

    # Calculate weighted binary cross-entropy loss for segmentation
    weights = ground_truths_flat * pos_weight + (1 - ground_truths_flat)
    bce_loss = F.binary_cross_entropy_with_logits(predictions_flat, ground_truths_flat, weight=weights)
    combined_loss = tversky_loss + bce_loss

    
    # Boundary Loss with Deep Supervision
    if boundary_predictions is not None and boundary_ground_truths is not None:
        if isinstance(boundary_predictions, torch.Tensor):
            boundary_predictions = [boundary_predictions]
        elif isinstance(boundary_predictions, list):
            pass
        else:
            raise TypeError("boundary_predictions must be a torch.Tensor or a list of torch.Tensor")

        if not isinstance(boundary_ground_truths, torch.Tensor):
            raise TypeError("boundary_ground_truths must be a torch.Tensor")

        boundary_loss = 0.0
        for idx, boundary_pred in enumerate(boundary_predictions):
            if boundary_pred.shape[-2:] != boundary_ground_truths.shape[-2:]:
                boundary_gt_resized = F.interpolate(
                    boundary_ground_truths.float(), 
                    size=boundary_pred.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )
            else:
                boundary_gt_resized = boundary_ground_truths.float()
            boundary_pred_flat = boundary_pred.view(-1)
            boundary_gt_flat = boundary_gt_resized.view(-1)
            boundary_bce = F.binary_cross_entropy_with_logits(boundary_pred_flat, boundary_gt_flat)
            boundary_loss += boundary_bce

        boundary_loss = boundary_loss / len(boundary_predictions)

        # Add boundary loss to the combined loss with weighting
        combined_loss += boundary_weight * boundary_loss

    return combined_loss


gc.collect()
torch.cuda.empty_cache()
