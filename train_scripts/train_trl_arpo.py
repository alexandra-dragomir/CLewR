from torch.utils.data import DataLoader
import json 
import os

        
from unsloth import FastLanguageModel # type: ignore
from datasets import load_dataset, Dataset # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig, CPOTrainer
import wandb
import pandas as pd
import argparse
from custom_losses_cpo import CustomCPOTrainer

def patch_tokenizer_chat_template(tokenizer):
    """
    Replace GemmaX2's incomplete chat template with the working Gemma2 template
    """
    working_chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    
    print("Original GemmaX2 chat template:")
    print(f"  {repr(tokenizer.chat_template)}")
    print("Replacing with working Gemma2 chat template...")
    
    tokenizer.chat_template = working_chat_template
    
    print("Chat template updated successfully!")
    
    return tokenizer

def load_model_with_fixed_tokenizer(model_name, max_seq_length):
    """
    Load model with Unsloth but fix tokenizer at runtime if needed
    """
    if "GemmaX2" in model_name or "gemmax2" in model_name.lower():
        print(f"Detecting GemmaX2 model: {model_name}. Creating fixed model+tokenizer...")
        
        temp_dir = "/root/DPO_translation_project/tmp/gemmax2_tmp_fixed2"
        if os.path.exists(temp_dir):
            print("Directory exists, skipping creation of complete model+tokenizer, loading from existing directory...")
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=temp_dir,  
                    max_seq_length=max_seq_length,
                    load_in_4bit=True,
                )
            except TimeoutError as e:
                print("Error loading model: ", e)
            
            print("Successfully loaded GemmaX2 with fixed tokenizer!")
               
        else:
            try:
                print("1. Loading original GemmaX2 model and tokenizer...")
                original_model = AutoModelForCausalLM.from_pretrained(model_name)
                original_tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                print("2. Patching tokenizer...")
                fixed_tokenizer = patch_tokenizer_chat_template(original_tokenizer)
                
                print("3. Saving complete fixed model...")
                original_model.save_pretrained(temp_dir)
                fixed_tokenizer.save_pretrained(temp_dir)
                
                print("4. Loading with Unsloth...")
                try:
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=temp_dir,  
                        max_seq_length=max_seq_length,
                        load_in_4bit=True,
                    )
                except TimeoutError as e:
                    print("Error loading model: ", e)
                
                print("Successfully loaded GemmaX2 with fixed tokenizer!")
                
            except Exception as e:
                print(f"Error loading model: {e}")
       
            
    else:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
            )
        except TimeoutError as e:
            print("Error loading model: ", e)
    
    return model, tokenizer

def main():
    
    parser = argparse.ArgumentParser(description="Train a DPO model with LoRA")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-2-9b-it", help="Name of the model to train")
    parser.add_argument("--train_dataset", type=str, default="/root/DPO_translation_project/TRL/CLewR/dataset/train+val/group2/xalma_pref_group2_train.json", help="Path to the training dataset")
    parser.add_argument("--eval_dataset", type=str, default="/root/DPO_translation_project/TRL/CLewR/dataset/train+val/group2/xalma_pref_group2_val.json", help="Path to the evaluation dataset")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for the model")
    parser.add_argument("--output_dir", type=str, default="/root/DPO_translation_project/TRL/CLewR/runs/", help="Directory to save the trained model and logs")
    parser.add_argument("--output_name", type=str, default="dpo_model", help="Name of the output model file")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--loss", type=str, default="ARPO", help="Loss function")
    parser.add_argument("--lr", type=float, default=5e-05, help="Learning rate")
    parser.add_argument("--eta", type=float, default=1.5, help="Eta value")
    parser.add_argument("--cpo_alpha", type=float, default=1.0, help="CPO alpha value")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta value")
    parser.add_argument("--eta_bleu", type=float, default=1.5, help="Eta bleu value")
    parser.add_argument("--eta_comet", type=float, default=6, help="Eta comet value")
    parser.add_argument("--z_alpha", type=float, default=0.5, help="Z alpha value")
    parser.add_argument("--z_beta", type=float, default=0.33, help="Z beta value")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval steps")
    args = parser.parse_args()
    
    # print configuration
    print("Configuration:")
    print(f"Model name: {args.model_name}")
    print(f"Train dataset: {args.train_dataset}")
    print(f"Eval dataset: {args.eval_dataset}")
    print(f"Output name: {args.output_name}")
    print(f"Loss: {args.loss}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of training epochs: {args.num_train_epochs}")
    print(f"Eta: {args.eta}")    
    print(f"Eta bleu: {args.eta_bleu}")
    print(f"Eta comet: {args.eta_comet}")
    print(f"Z alpha: {args.z_alpha}")
    print(f"Z beta: {args.z_beta}")
    print(f"Save steps: {args.save_steps}")
    
    train_dataset_json = json.load(open(args.train_dataset, "r"))
    eval_dataset_json = json.load(open(args.eval_dataset, "r"))
    
    train_dataset_json = train_dataset_json[:100]
    eval_dataset_json = eval_dataset_json[:100]
    
    print(f"Loaded train dataset with length: {len(train_dataset_json)} from {args.train_dataset}")
    print(f"Loaded eval dataset with length: {len(eval_dataset_json)} from {args.eval_dataset}")

    train_dataset = Dataset.from_list(train_dataset_json)
    eval_dataset = Dataset.from_list(eval_dataset_json)

    model, tokenizer = load_model_with_fixed_tokenizer(args.model_name, args.max_length)
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,  
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=2*args.lora_r,
        lora_dropout=0,  
        bias="none",  
        use_gradient_checkpointing=True,
        random_state=1024,
    )
    print("Model and tokenizer loaded successfully.")
    
    training_args = CPOConfig(
        output_dir=args.output_dir+args.output_name,
        loss_type=args.loss,
        bf16=True,
        num_train_epochs=args.num_train_epochs,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cpo_alpha=args.cpo_alpha,
        learning_rate=args.lr,
    )

    trainer = CustomCPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        shuffle=False,
        eta=args.eta,
        eta_comet=args.eta_comet,
        eta_bleu=args.eta_bleu,
        z_alpha=args.z_alpha,
        z_beta=args.z_beta,
    )
    
    
    print("Trainer initialized successfully.")
    
    run_name = args.output_name
    wandb.init(project="huggingface", name=run_name)
    
    print("Starting training...")

    trainer.train()
    print("Training completed successfully.")
    
    print("Training logs saved to wandb in real-time.")

if __name__ == "__main__":
    main()
