"""
Custom DPO Trainer with additional loss functions.

Inherits from DPOTrainer and overrides the dpo_loss method to add custom losses.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union
from trl import DPOTrainer


class CustomDPOTrainer(DPOTrainer):
    """
    Custom DPO Trainer that extends the base DPOTrainer with additional loss functions.
    """
    
    def __init__(self, *args, lambda_reg: float = 50.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = lambda_reg
    
    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        loss_type: str = "sigmoid",
        model_output: dict[str, torch.FloatTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        device = self.accelerator.device
        
        if loss_type == "dpop":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            reg_term = torch.clamp(
                ref_rejected_logps - rejected_logps, 
                min=0.0
            )
            losses = (
                -F.logsigmoid(self.beta * logits - self.lambda_reg * reg_term) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            
            # Compute rewards
            chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
            rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()
            
            return losses, chosen_rewards, rejected_rewards
        
        # Fall back to the original DPO loss implementation
        return super().dpo_loss(
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            loss_type=loss_type,
            model_output=model_output,
        )
