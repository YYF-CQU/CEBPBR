import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import (
    F1Score,
    Precision,
    Recall,
    Accuracy
)
from torchmetrics import Metric
from hausdorff import hausdorff_distance
import numpy as np
from scipy.ndimage import convolve, center_of_mass, distance_transform_edt as dtedt
import gc
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define Custom Metrics
class MDice(Metric):
    """
    Modified Dice Coefficient Metric
    """
    def __init__(self, threshold=0.5, reduction='mean', **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.reduction = reduction
        self.add_state('dice_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, targets):
        # preds: [B, 1, H, W]
        # targets: [B, 1, H, W]
        preds = (preds > self.threshold).float()
        targets = (targets > self.threshold).float()
        intersection = (preds * targets).sum(dim=(1,2,3))
        union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + 1e-15
        dice = (2 * intersection) / union
        self.dice_sum += dice.sum()
        self.num_samples += preds.size(0)

    def compute(self):
        return self.dice_sum / self.num_samples


class MIoU(Metric):
    """
    Modified Intersection over Union (IoU) Metric
    """
    def __init__(self, threshold=0.5, reduction='mean', **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.reduction = reduction
        self.add_state('iou_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, targets):
        # preds: [B, 1, H, W]
        # targets: [B, 1, H, W]
        preds = (preds > self.threshold).float()
        targets = (targets > self.threshold).float()
        intersection = (preds * targets).sum(dim=(1,2,3))
        union = (preds + targets).sum(dim=(1,2,3)) - intersection + 1e-15
        iou = intersection / union
        self.iou_sum += iou.sum()
        self.num_samples += preds.size(0)

    def compute(self):
        return self.iou_sum / self.num_samples


class WFbetaMetric(Metric):
    """
    Weighted F-beta Metric
    Reference: How to Evaluate Foreground Maps? (CVPR2014)
    """
    def __init__(self, beta=1, reduction='mean', **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.eps = 1e-12
        self.add_state('wfb_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def _gaussian_distribution(self, x, mu, sigma):
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) / sigma)**2 / 2)

    def _generate_gaussian_kernel(self, size, sigma=1.0, mu=0.0):
        kernel_1d = np.linspace(-(size//2), size//2, size)
        kernel_1d = self._gaussian_distribution(kernel_1d, mu, sigma)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.max()
        return kernel_2d

    def update(self, y_mask, y_pred):
        y_mask_np = (y_mask > 0.5).cpu().numpy()   # Boolean array
        y_pred_np = (y_pred > 0.5).cpu().numpy()   # Boolean array
        batch_size = y_mask_np.shape[0]
        wfb_batch = []
        K = self._generate_gaussian_kernel(size=7, sigma=5.0)
        
        for i in range(batch_size):
            # Convert mask and prediction to appropriate types
            mask_bool = y_mask_np[i, 0, :, :].astype(bool)      
            pred_bool = y_pred_np[i, 0, :, :].astype(bool)      
            mask_float = mask_bool.astype(float)                
            pred_float = pred_bool.astype(float)               

            
            try:
                Dst, Idxt = dtedt(~mask_bool, return_indices=True)
            except Exception as e:
                raise ValueError(f"Error in distance_transform_edt: {e}")

            E = np.abs(pred_float - mask_float)  

            # Compute Et: Adjusted E using the distance transform indices
            Et = np.copy(E)
            Et[mask_float == 0] = E[Idxt[0][mask_float == 0], Idxt[1][mask_float == 0]]

            # Convolve Et with Gaussian kernel
            EA = convolve(Et, weights=K, mode='constant', cval=0)

            # Compute MIN_E_EA using mask_bool instead of mask_float
            MIN_E_EA = np.where(mask_bool & (EA < E), EA, E)

            # Compute B
            B = np.where(mask_float == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(mask_float))

            # Compute Ew
            Ew = MIN_E_EA * B

            # Compute TPw and FPw
            TPw = mask_float.sum() - Ew[mask_float == 1].sum()
            FPw = Ew[mask_float == 0].sum()

            # Compute R and P
            R = 1 - np.mean(Ew[mask_float == 1]) if mask_float.sum() > 0 else 1.0
            P = TPw / (self.eps + TPw + FPw)

            # Compute wFb
            wfb = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)
            wfb_batch.append(wfb)
        
        # Convert wfb_batch to a tensor and sum elements
        wfb_tensor = torch.tensor(wfb_batch, dtype=torch.float32, device=self.device)
        wfb_sum_batch = wfb_tensor.sum()
        self.wfb_sum += wfb_sum_batch

        self.num_samples += batch_size

    def compute(self):
        return self.wfb_sum / self.num_samples


class SMeasure(Metric):
    """
    Structure Measure Metric
    Reference: Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    """
    def __init__(self, alpha=0.5, reduction='mean', **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.add_state('s_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def _object(self, inp1, inp2):
        if not inp2.any():
            return 0.0
        x = np.mean(inp1[inp2])
        sigma_x = np.std(inp1[inp2])
        score = 2 * x / (x**2 + 1 + sigma_x + 1e-8)
        return score

    def _ssim(self, SM, GT):
        h, w = SM.shape
        N = h * w
        if N <= 1:
            return 0.0  

        x = SM.mean()
        y = GT.mean()
        sigma_x = SM.var()
        sigma_y = GT.var()
        sigma_xy = ((SM - x) * (GT - y)).sum() / (N - 1) if N > 1 else 0

        alpha_val = 4 * x * y * sigma_xy
        beta_val = (x**2 + y**2) * (sigma_x + sigma_y)
        if beta_val == 0:
            return 1.0 if alpha_val == 0 else 0.0
        score = alpha_val / (beta_val + 1e-8)
        return score

    def _divideGT(self, GT, x, y):
        h, w = GT.shape
        area = h * w
        UL = GT[0:y, 0:x]
        UR = GT[0:y, x:w]
        LL = GT[y:h, 0:x]
        LR = GT[y:h, x:w]
        w1 = (x * y) / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area
        return UL, UR, LL, LR, w1, w2, w3, w4

    def _divideSM(self, SM, x, y):
        h, w = SM.shape
        UL = SM[0:y, 0:x]
        UR = SM[0:y, x:w]
        LL = SM[y:h, 0:x]
        LR = SM[y:h, x:w]
        return UL, UR, LL, LR

    def s_region(self, GT, SM):
        com = center_of_mass(GT)
        if np.isnan(com[0]) or np.isnan(com[1]):
            com = (GT.shape[0] // 2, GT.shape[1] // 2)
        y_center = int(round(com[0])) + 1
        x_center = int(round(com[1])) + 1
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(GT, x_center, y_center)
        sm1, sm2, sm3, sm4 = self._divideSM(SM, x_center, y_center)
        score1 = self._ssim(sm1, gt1)
        score2 = self._ssim(sm2, gt2)
        score3 = self._ssim(sm3, gt3)
        score4 = self._ssim(sm4, gt4)
        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def s_object_similarity(self, SM, GT):
        fg = SM * GT
        bg = (1 - SM) * (1 - GT)
        u = GT.mean()
        GT_bool = GT.astype(bool)
        
        if GT_bool.any() and (~GT_bool).any():
            obj_score = self._object(fg, GT_bool)
            bg_score = self._object(bg, ~GT_bool)
            return u * obj_score + (1 - u) * bg_score
        elif GT_bool.any():
            obj_score = self._object(fg, GT_bool)
            return u * obj_score
        elif (~GT_bool).any():
            bg_score = self._object(bg, ~GT_bool)
            return (1 - u) * bg_score
        else:
            return 0.0

    def update(self, y_mask, y_pred):
        y_mask_np = (y_mask > 0.5).cpu().numpy()
        y_pred_np = (y_pred > 0.5).cpu().numpy()
        batch_size = y_mask_np.shape[0]
        s_batch = []
        for i in range(batch_size):
            GT = y_mask_np[i, 0, :, :]
            SM = y_pred_np[i, 0, :, :]
            u = GT.mean()
            if u == 0:
                score = 1 - SM.mean()
            elif u == 1:
                score = SM.mean()
            else:
                obj = self.s_object_similarity(SM, GT)
                reg = self.s_region(GT, SM)
                score = self.alpha * obj + (1 - self.alpha) * reg
            score = 0.0 if np.isnan(score) else score
            s_batch.append(score)
        # Convert s_batch to a tensor and sum its elements
        s_tensor = torch.tensor(s_batch, dtype=torch.float32, device=self.device)
        s_sum_batch = s_tensor.sum()
        # Accumulate the sum
        self.s_sum += s_sum_batch
        # Accumulate the number of samples
        self.num_samples += batch_size

    def compute(self):
        return self.s_sum / self.num_samples

    
    

class EMeasure(Metric):
    """
    Enhanced Alignment Measure for Binary Foreground Map Evaluation
    Reference: Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    """
    def __init__(self, reduction='mean', **kwargs):
        super().__init__(**kwargs)
        self.add_state('em_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def AlignmentTerm(self, dFM, dy_mask):
        mu_FM = np.mean(dFM)
        mu_y_mask = np.mean(dy_mask)
        align_FM = dFM - mu_FM
        align_y_mask = dy_mask - mu_y_mask
        align_Matrix = 2. * (align_y_mask * align_FM) / (align_y_mask**2 + align_FM**2 + 1e-8)
        return align_Matrix

    def EnhancedAlignmentTerm(self, align_Matrix):
        enhanced = (align_Matrix + 1) ** 2 / 4
        return enhanced

    def update(self, y_mask, y_pred):
        y_mask_np = (y_mask > 0.5).cpu().numpy()
        y_pred_np = (y_pred > 0.5).cpu().numpy()
        batch_size = y_mask_np.shape[0]
        em_batch = []
        for i in range(batch_size):
            mask = y_mask_np[i,0,:,:]
            pred = y_pred_np[i,0,:,:]
            th = 2 * pred.mean()
            th = min(th, 1)
            FM = (pred >= th).astype(bool)
            mask_bool = mask.astype(bool)
            if mask.sum() == 0:
                enhanced_matrix = 1.0 - FM.astype(float)
            elif (~mask_bool).sum() == 0:
                enhanced_matrix = FM.astype(float)
            else:
                dy_mask = mask.astype(float)
                dFM = FM.astype(float)
                align_matrix = self.AlignmentTerm(dFM, dy_mask)
                enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)
            w, h = mask.shape
            score = enhanced_matrix.sum() / (w * h - 1 + 1e-8)
            em_batch.append(score)
        # Convert em_batch to a tensor and sum its elements
        em_tensor = torch.tensor(em_batch, dtype=torch.float32, device=self.device)
        em_sum_batch = em_tensor.sum()

        # Accumulate the sum
        self.em_sum += em_sum_batch
        self.num_samples += batch_size

    def compute(self):
        return self.em_sum / self.num_samples



# Relative Area Difference (RAD)

def relative_area_difference(predictions, targets, threshold=0.5):
    """
    Computes the Relative Area Difference (RAD) between predictions and targets.

    RAD = |Area_pred - Area_gt| / Area_gt

    Arguments:
    - predictions: The predicted logits from the model for segmentation.
    - targets: The ground truth binary labels for segmentation.
    - threshold: Threshold to binarize the predictions.

    Returns:
    - Mean RAD over the batch.
    """
    pred = (torch.sigmoid(predictions) > threshold).float()
    gt = targets.float()

    area_pred = pred.view(pred.size(0), -1).sum(dim=1)
    area_gt = gt.view(gt.size(0), -1).sum(dim=1)

    rad = torch.abs(area_pred - area_gt) / (area_gt + 1e-8)
    return rad.mean()


# Define MedicalSegmentationModel with Updated Metrics

class MedicalSegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'CEBPBR',
        num_classes: int = 1,
        init_lr: float = 0.001,
        optimizer_name: str = "Adam",
        weight_decay: float = 1e-4,
        use_scheduler: bool = False,
        scheduler_name: str = "multistep_lr",
        num_epochs: int = 20,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Initialize the model
        self.model = CEBPBR()
        
        
        # Metrics for Training
        self.train_mdice = MDice()
        self.train_miou = MIoU()
        self.train_wfb = WFbetaMetric()
        self.train_smeasure = SMeasure()
        self.train_emasure = EMeasure()
        self.train_f1 = F1Score(task="binary", threshold=0.5)
        self.train_mae = torch.nn.L1Loss()
        self.train_precision = Precision(task="binary", threshold=0.5)
        self.train_recall = Recall(task="binary", threshold=0.5)
        self.train_accuracy = Accuracy(task="binary", threshold=0.5)
        
        
        # Metrics for Validation
        self.val_mdice = MDice()
        self.val_miou = MIoU()
        self.val_wfb = WFbetaMetric()
        self.val_smeasure = SMeasure()
        self.val_emasure = EMeasure()
        self.val_f1 = F1Score(task="binary", threshold=0.5)
        self.val_mae = torch.nn.L1Loss()
        self.val_precision = Precision(task="binary", threshold=0.5)
        self.val_recall = Recall(task="binary", threshold=0.5)
        self.val_accuracy = Accuracy(task="binary", threshold=0.5)

    def compute_hausdorff_distance(self, predicted, target):
        """
        Computes the Hausdorff Distance between predicted and target masks.
        """
        predicted = (torch.sigmoid(predicted) > 0.5).cpu().numpy()
        target = target.cpu().numpy()

        hausdorff_values = []
        for pred, gt in zip(predicted, target):
            pred = pred.squeeze().astype(np.float32)
            gt = gt.squeeze().astype(np.float32)

            pred_points = np.argwhere(pred > 0)
            gt_points = np.argwhere(gt > 0)

            if len(pred_points) == 0 and len(gt_points) == 0:
                hausdorff_values.append(0.0)  # Both masks are empty
            elif len(pred_points) == 0 or len(gt_points) == 0:
                hausdorff_values.append(1e6)  
            else:
                hausdorff_values.append(hausdorff_distance(pred_points, gt_points))

        hausdorff_values = [
            hv if np.isfinite(hv) else 1e6 for hv in hausdorff_values
        ]
        return torch.tensor(hausdorff_values).mean()

    def forward(self, data):
        """
        Forward pass through the model.

        Returns:
        - segmentation_output: The predicted segmentation logits.
        - refined_boundary: The refined boundary logits.
        - aux_boundary_outputs: List of auxiliary boundary logits from deep supervision.
        """
        segmentation_output, refined_boundary, aux_boundary_outputs = self.model(data)
        return segmentation_output, refined_boundary, aux_boundary_outputs

    def training_step(self, batch, batch_idx):
        data, target = batch

        # Add a channel dimension to the target to match the predictions
        target = target.unsqueeze(1)

        # Forward pass
        segmentation_output, boundary_output, aux_boundary_outputs = self(data)

        # Combine main boundary and auxiliary boundaries
        boundary_predictions = [boundary_output] + aux_boundary_outputs
        loss = tversky_bce_loss(
            predictions=segmentation_output, 
            ground_truths=target, 
            boundary_predictions=boundary_predictions, 
            boundary_ground_truths=target,
            alpha=0.5, 
            beta=0.5, 
            pos_weight=1.0, 
            boundary_weight=0.5
        )

        # Compute Metrics
        preds = segmentation_output
        targets = target

        # Update training metrics
        self.train_mdice.update(preds, targets)
        self.train_miou.update(preds, targets)
        self.train_wfb.update(preds, targets)
        self.train_smeasure.update(preds, targets)
        self.train_emasure.update(preds, targets)

        # Existing torchmetrics
        f1 = self.train_f1(preds, targets)
        mae = self.train_mae(torch.sigmoid(segmentation_output), targets)
        precision = self.train_precision(preds, targets)
        recall = self.train_recall(preds, targets)
        accuracy = self.train_accuracy(preds, targets)
        rad = relative_area_difference(segmentation_output, target)
        hausdorff = self.compute_hausdorff_distance(segmentation_output, target)

        # Log the metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/mDice", self.train_mdice, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/mIoU", self.train_miou, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/wFb", self.train_wfb, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/S-measure", self.train_smeasure, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/E-measure", self.train_emasure, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/f1_score", f1, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/mae", mae, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/precision", precision, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/recall", recall, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/rad", rad, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/hausdorff", hausdorff, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch

        # Add a channel dimension to the target to match the predictions
        target = target.unsqueeze(1)

        # Forward pass
        segmentation_output, boundary_output, aux_boundary_outputs = self(data)

        # Combine main boundary and auxiliary boundaries
        boundary_predictions = [boundary_output] + aux_boundary_outputs
        loss = tversky_bce_loss(
            predictions=segmentation_output, 
            ground_truths=target, 
            boundary_predictions=boundary_predictions, 
            boundary_ground_truths=target,
            alpha=0.5, 
            beta=0.5, 
            pos_weight=1.0, 
            boundary_weight=0.5
        )

        # Compute Metrics
        preds = segmentation_output
        targets = target

        # Update validation metrics
        self.val_mdice.update(preds, targets)
        self.val_miou.update(preds, targets)
        self.val_wfb.update(preds, targets)
        self.val_smeasure.update(preds, targets)
        self.val_emasure.update(preds, targets)

        # Existing torchmetrics
        f1 = self.val_f1(preds, targets)
        mae = self.val_mae(torch.sigmoid(segmentation_output), targets)
        precision = self.val_precision(preds, targets)
        recall = self.val_recall(preds, targets)
        accuracy = self.val_accuracy(preds, targets)
        rad = relative_area_difference(segmentation_output, target)
        hausdorff = self.compute_hausdorff_distance(segmentation_output, target)

        # Log the metrics
        self.log("valid/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/mDice", self.val_mdice, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/mIoU", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/wFb", self.val_wfb, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/S-measure", self.val_smeasure, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/E-measure", self.val_emasure, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/f1_score", f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/precision", precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/recall", recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/rad", rad, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/hausdorff", hausdorff, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            self.parameters(),
            lr=self.hparams.init_lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.use_scheduler:
            if self.hparams.scheduler_name == "multistep_lr":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, 
                    milestones=[self.hparams.num_epochs // 2], 
                    gamma=0.1
                )
            elif self.hparams.scheduler_name == "step_lr":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=10, 
                    gamma=0.1
                )
            else:
                raise ValueError(f"Unsupported scheduler: {self.hparams.scheduler_name}")
            return [optimizer], [scheduler]
        return optimizer


pl.seed_everything(42, workers=True)

# Initialize custom model for binary segmentation.
model = MedicalSegmentationModel(
    model_name="CEBPBR",
    num_classes=1,  
    init_lr=0.001,
    optimizer_name="AdamW",
    weight_decay=1e-4,
    use_scheduler=True,
    scheduler_name="multistep_lr",  
    num_epochs=20,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


x = torch.randn((1, 3, 192, 192)).to(device)  
# Perform a forward pass through the model
# Unpack all three outputs
segmentation_output, boundary_output, aux_boundary_outputs = model(x)

print(f"Segmentation Output Shape: {segmentation_output.shape}")  

print(f"Boundary Output Shape: {boundary_output.shape}")  

print(f"Auxiliary Boundary Outputs: {[aux.shape for aux in aux_boundary_outputs]}")
