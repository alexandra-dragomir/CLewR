"""
Custom CPO Trainer with additional loss functions.

Inherits from CPOTrainer and overrides the cpo_loss method to add custom losses.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Literal, Union
from trl import CPOTrainer


class CustomCPOTrainer(CPOTrainer):
    """
    Custom CPO Trainer that extends the base CPOTrainer with additional loss functions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = kwargs.get("eta", 1.5)
        self.eta_bleu = kwargs.get("eta_bleu", 1.5)
        self.eta_comet = kwargs.get("eta_comet", 6)
        self.z_alpha = kwargs.get("z_alpha", 0.5)
        self.z_beta = kwargs.get("z_beta", 0.33)
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
            chosen_len, 
            rejected_len
        ) = forward_output[:7]
        if self.aux_loss_enabled:
            aux_loss = forward_output[7]


        losses, chosen_rewards, rejected_rewards, z, tau = self.cpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_len,
            rejected_len,
            batch  
        )

        loss = losses.mean() + self.cpo_alpha * policy_nll_loss
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(policy_rejected_logps).detach().mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(policy_chosen_logps).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(policy_rejected_logits.detach().mean()).mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(policy_chosen_logits.detach().mean()).mean().item()
        )
        metrics[f"{prefix}nll_loss"] = self.accelerator.gather_for_metrics(policy_nll_loss).detach().mean().item()
        
        # Only log z and tau if they are tensors (from custom loss functions, not original CPO sigmoid loss)
        if isinstance(z, torch.Tensor) and hasattr(z, 'mean'):
            metrics[f"{prefix}z"] = self.accelerator.gather_for_metrics(z).mean().item()
        if isinstance(tau, torch.Tensor) and hasattr(tau, 'mean'):
            metrics[f"{prefix}tau"] = self.accelerator.gather_for_metrics(tau).mean().item()
        
        metrics[f"{prefix}policy_chosen_logps_normalized"] = self.accelerator.gather_for_metrics(policy_chosen_logps / chosen_len).mean().item()
        metrics[f"{prefix}policy_rejected_logps_normalized"] = self.accelerator.gather_for_metrics(policy_rejected_logps / rejected_len).mean().item()

        if self.aux_loss_enabled:
            loss += self.aux_loss_coef * aux_loss

        return loss, metrics
    
    def cpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_chosen_logits: Optional[torch.FloatTensor] = None,
        policy_rejected_logits: Optional[torch.FloatTensor] = None,
        chosen_len: Optional[torch.LongTensor] = None,
        rejected_len: Optional[torch.LongTensor] = None,
        batch: Optional[dict[str, Union[list, torch.LongTensor]]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the CPO loss for a batch of policy log probabilities.
        
        This method extends the base cpo_loss to support additional custom loss types.
        Add your custom loss implementations in the elif branches below.
        
        Args:
            policy_chosen_logps: Log probabilities of the policy model for chosen responses
            policy_rejected_logps: Log probabilities of the policy model for rejected responses
            policy_chosen_logits: Logits of the policy model for chosen responses (optional)
            policy_rejected_logits: Logits of the policy model for rejected responses (optional) 
            chosen_len: Length of chosen sequences (optional)
            rejected_len: Length of rejected sequences (optional)
            batch: The batch of data being processed (optional)
            
        Returns:
            Tuple of (losses, chosen_rewards, rejected_rewards, z, tau)
        """
        
         #  ======================== ARPO LOSS ========================
        if self.loss_type == "ARPO":
            
            z = (policy_chosen_logps / chosen_len - policy_rejected_logps / rejected_len).abs()
            arg = torch.clamp(self.eta * z, max=30.0)
            tau = torch.expm1(arg)       
            tau = torch.clamp(tau, max=1.0)
            logits = (policy_chosen_logps - tau * policy_rejected_logps).to(self.accelerator.device)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing 
            )
        
        elif self.loss_type == "ARPO_z_bleu":
            batch_bleu = torch.tensor(batch["bleu"], dtype=torch.float32, device=self.accelerator.device)
            z = 1 - batch_bleu / 100
            arg = torch.clamp(self.eta_bleu * z, max=30.0)
            tau = torch.expm1(arg)        
            tau = torch.clamp(tau, max=1.0)
            logits = (policy_chosen_logps - tau * policy_rejected_logps).to(self.accelerator.device)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing  
            )
        elif self.loss_type == "ARPO_z_comet":
            batch_comet = torch.tensor(batch["comet"], dtype=torch.float32, device=self.accelerator.device)
            z = 1 - batch_comet / 100
            arg = torch.clamp(self.eta_comet * z, max=30.0)
            tau = torch.expm1(arg)        
            tau = torch.clamp(tau, max=1.0)
            logits = (policy_chosen_logps - tau * policy_rejected_logps).to(self.accelerator.device)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing 
            )
        elif self.loss_type == "ARPO_z_bleu_comet":
            batch_bleu = torch.tensor(batch["bleu"], dtype=torch.float32, device=self.accelerator.device)
            batch_comet = torch.tensor(batch["comet"], dtype=torch.float32, device=self.accelerator.device)
            z_bleu = 1 - batch_bleu / 100
            z_comet = 1 - batch_comet / 100
            z = self.z_alpha * self.eta_bleu * z_bleu + (1 - self.z_alpha) * self.eta_comet * z_comet    
            arg = torch.clamp(z, max=30.0)
            tau = torch.expm1(arg)       
            tau = torch.clamp(tau, max=1.0)
            logits = (policy_chosen_logps - tau * policy_rejected_logps).to(self.accelerator.device)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing    
            )
        elif self.loss_type == "ARPO_z_z_bleu":
            batch_bleu = torch.tensor(batch["bleu"], dtype=torch.float32, device=self.accelerator.device)
            z_initial = (policy_chosen_logps / chosen_len - policy_rejected_logps / rejected_len).abs()
            z_bleu = 1 - batch_bleu / 100
            z = self.z_alpha * self.eta * z_initial + (1 - self.z_alpha) * self.eta_bleu * z_bleu
            arg = torch.clamp(z, max=30.0)
            tau = torch.expm1(arg)        
            tau = torch.clamp(tau, max=1.0)
            logits = (policy_chosen_logps - tau * policy_rejected_logps).to(self.accelerator.device)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing    
            )
        elif self.loss_type == "ARPO_z_z_comet":
            batch_comet = torch.tensor(batch["comet"], dtype=torch.float32, device=self.accelerator.device)
            z_initial = (policy_chosen_logps / chosen_len - policy_rejected_logps / rejected_len).abs()
            z_comet = 1 - batch_comet / 100
            z = self.z_alpha * self.eta * z_initial + (1 - self.z_alpha) * self.eta_comet * z_comet
            arg = torch.clamp(z, max=30.0)
            tau = torch.expm1(arg)        
            tau = torch.clamp(tau, max=1.0)
            logits = (policy_chosen_logps - tau * policy_rejected_logps).to(self.accelerator.device)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing    
            )
        elif self.loss_type == "ARPO_z_z_bleu_z_comet":
            batch_bleu = torch.tensor(batch["bleu"], dtype=torch.float32, device=self.accelerator.device)
            batch_comet = torch.tensor(batch["comet"], dtype=torch.float32, device=self.accelerator.device)
            z_initial = (policy_chosen_logps / chosen_len - policy_rejected_logps / rejected_len).abs()
            z_bleu = 1 - batch_bleu / 100
            z_comet = 1 - batch_comet / 100
            z = self.z_alpha * self.eta * z_initial + self.z_beta * self.eta_bleu * z_bleu + (1 - self.z_alpha - self.z_beta) * self.eta_comet * z_comet
            arg = torch.clamp(z, max=30.0)
            tau = torch.expm1(arg)        
            tau = torch.clamp(tau, max=1.0)
            logits = (policy_chosen_logps - tau * policy_rejected_logps).to(self.accelerator.device)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing 
            )
        else:
            # Fall back to the original CPO loss implementation
            return super().cpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                policy_chosen_logits=policy_chosen_logits,
                policy_rejected_logits=policy_rejected_logits,
                chosen_len=chosen_len,
                rejected_len=rejected_len,
                
            )
        
        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()
        
        return losses, chosen_rewards, rejected_rewards, z, tau