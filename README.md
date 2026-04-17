# CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning - ACL 2026 (official repo)

## Abstract

Large language models (LLMs) have demonstrated competitive performance in zero-shot multilingual machine translation (MT). Some follow-up works further improved MT performance via preference optimization, but they leave a key aspect largely underexplored: the order in which data samples are given during training. We address this topic by integrating curriculum learning into various state-of-the-art preference optimization algorithms to boost MT performance. We introduce **CLewR** (**C**urriculum **Le**arning **w**ith **R**estarts), a novel strategy that reiterates easy-to-hard curriculum multiple times during training to effectively mitigate the catastrophic forgetting of easy examples. We demonstrate consistent gains across several model families (Gemma2, Qwen2.5, Llama3.1) and preference optimization techniques. We publicly release our code at [https://github.com/alexandra-dragomir/CLewR](https://github.com/alexandra-dragomir/CLewR).

Released under the [MIT License](LICENSE).

---

## Quick Start

### DPO Training (with DPOP loss)
```bash
python train_scripts/train_trl_dpo.py \
    --model_name unsloth/gemma-2-9b-it \
    --train_dataset path/to/train.json \
    --eval_dataset path/to/val.json \
    --output_name my_dpo_run \
    --loss dpop \
    --beta 0.1 \
    --num_train_epochs 3
```

### ARPO Training (CPO-based)
```bash
python train_scripts/train_trl_arpo.py \
    --model_name unsloth/gemma-2-9b-it \
    --train_dataset path/to/train.json \
    --eval_dataset path/to/val.json \
    --output_name my_arpo_run \
    --loss ARPO \
    --beta 0.1 \
    --eta 1.5 \
    --num_train_epochs 3
```

## Hyperparameters

### Common Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `unsloth/gemma-2-9b-it` | Model to train |
| `--train_dataset` | - | Path to training JSON |
| `--eval_dataset` | - | Path to evaluation JSON |
| `--output_dir` | `runs/` | Output directory |
| `--output_name` | `dpo_model` | Run name (used for wandb + save path) |
| `--max_length` | `2048` | Max sequence length |
| `--num_train_epochs` | `3` | Training epochs |
| `--per_device_train_batch_size` | `4` | Batch size |
| `--gradient_accumulation_steps` | `2` | Gradient accumulation |
| `--lora_r` | `64` | LoRA rank |
| `--beta` | `0.1` | DPO/CPO beta |
| `--save_steps` | `500` | Checkpoint interval |
| `--eval_steps` | `500` | Evaluation interval |

### DPO-specific (`train_trl_dpo.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loss` | `dpop` | Loss type: `sigmoid`, `ipo`, `dpop`, etc. |

### ARPO-specific (`train_trl_arpo.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loss` | `ARPO` | Loss type (see below) |
| `--lr` | `5e-05` | Learning rate |
| `--eta` | `1.5` | Eta for log-prob based z |
| `--eta_bleu` | `1.5` | Eta for BLEU-based z |
| `--eta_comet` | `6` | Eta for COMET-based z |
| `--z_alpha` | `0.5` | Weight for first z component |
| `--z_beta` | `0.33` | Weight for second z component |
| `--cpo_alpha` | `1.0` | NLL loss weight |

### ARPO Loss Types
- `ARPO` - Uses normalized log-prob difference
- `ARPO_z_bleu` - Uses BLEU score
- `ARPO_z_comet` - Uses COMET score  
- `ARPO_z_bleu_comet` - Combines BLEU + COMET
- `ARPO_z_z_bleu` - Combines log-prob + BLEU
- `ARPO_z_z_comet` - Combines log-prob + COMET
- `ARPO_z_z_bleu_z_comet` - Combines all three

## Dataset Format
JSON with preference pairs:
```json
[
    {
        "prompt": "Translate this from English to Italian:\n English: The comments come after various parts of the U.S. government have made AI announcements, even as the U.S. overall lacks a formal AI strategy.\n Italian:",
        "chosen": "I commenti arrivano dopo che varie parti del governo degli Stati Uniti hanno fatto annunci sull'intelligenza artificiale, nonostante negli Stati Uniti nel complesso manchi una strategia formale sull'IA.",
        "rejected": "I commenti arrivano dopo che varie parti del governo degli Stati Uniti hanno fatto annunci sull'IA, anche se negli Stati Uniti in generale manca una strategia formale sull'IA.",
        "bleu": 58.02,
        "comet": 91.43,
        "source_lang": "en",
        "target_lang": "it"
    }
]
```
`bleu` and `comet` fields required only for `ARPO_z_*` variants.

