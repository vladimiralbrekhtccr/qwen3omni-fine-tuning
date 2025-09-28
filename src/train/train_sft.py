import os
import sys
import json
import logging
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,  # Use Auto classes for better compatibility
    AutoProcessor,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed
)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_path_or_name: str = field(
        metadata={"help": "Path or Hugging Face name of the Qwen3-Omni model and processor."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for training"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "The rank of the LoRA matrices."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha parameter for LoRA scaling."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability for LoRA layers."}
    )

@dataclass
class DataArguments:
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain one or more .jsonl files."}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )

class TextSFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        logger.info(f"Loading data from {data_path}...")
        jsonl_files = sorted(Path(data_path).glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"No .jsonl files found in {data_path}")
        
        dsets = [load_dataset("json", data_files=str(f), split="train") for f in jsonl_files]
        self.raw_ds = concatenate_datasets(dsets)
        
        logger.info(f"Data loading complete. Loaded {len(self.raw_ds)} samples.")

    def __len__(self):
        return len(self.raw_ds)

    def __getitem__(self, i):
        item = self.raw_ds[i]
        
        # Prepare the conversation in the chat template format
        conversation = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ]
        
        # Apply the chat template to get a formatted string
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize the text
        model_inputs = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,  # Padding will be done in the collator
            return_tensors=None
        )
        
        # Set up labels for language modeling
        # Copy input_ids to labels
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

class DataCollatorForCausalLM:
    """Custom data collator for causal language modeling"""
    
    def __init__(self, tokenizer, max_length=2048, model_type="standard"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Find max length in batch for efficient padding
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)
        
        # Pad sequences
        padded_inputs = []
        padded_labels = []
        attention_masks = []
        
        for inp, lab in zip(input_ids, labels):
            # Truncate if necessary
            inp = inp[:max_len]
            lab = lab[:max_len]
            
            # Calculate padding needed
            padding_len = max_len - len(inp)
            
            # Pad input_ids with pad_token_id
            padded_inp = inp + [self.tokenizer.pad_token_id] * padding_len
            
            # Pad labels with -100 (ignore index for loss)
            padded_lab = lab + [-100] * padding_len
            
            # Create attention mask
            attn_mask = [1] * len(inp) + [0] * padding_len
            
            padded_inputs.append(padded_inp)
            padded_labels.append(padded_lab)
            attention_masks.append(attn_mask)
        
        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
        }

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set training arguments
    training_args.optim = "adamw_torch"
    training_args.remove_unused_columns = False
    training_args.label_names = ["labels"]  # Explicitly set label names

    # Setup logging
    if training_args.should_log:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, bf16 training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    set_seed(training_args.seed)

    logger.info("Loading tokenizer...")
    # Try to load the processor/tokenizer
    try:
        from transformers import Qwen3OmniMoeProcessor
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_args.model_path_or_name, 
            trust_remote_code=True
        )
        tokenizer = processor.tokenizer
    except:
        # Fallback to AutoTokenizer if processor fails
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_path_or_name,
            trust_remote_code=True
        )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading model...")
    # Load the correct model component for training
    try:
        # First try to load the full model to check architecture
        from transformers import Qwen3OmniMoeForConditionalGeneration
        full_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_args.model_path_or_name,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            trust_remote_code=True,
        )
        
        # Extract the thinker component which has the forward method
        if hasattr(full_model, 'thinker'):
            logger.info("Using thinker component from Qwen3OmniMoeForConditionalGeneration")
            model = full_model.thinker
        else:
            logger.warning("No thinker component found, using full model")
            model = full_model
            
    except Exception as e:
        logger.warning(f"Failed to load Qwen3OmniMoeForConditionalGeneration: {e}")
        logger.info("Trying to load thinker model directly...")
        try:
            # Try loading the thinker model directly
            from transformers import Qwen3OmniMoeThinkerForConditionalGeneration
            model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                model_args.model_path_or_name,
                torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
                trust_remote_code=True,
            )
        except:
            logger.info("Falling back to AutoModelForCausalLM...")
            # Final fallback to AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_path_or_name,
                torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
                trust_remote_code=True,
            )
    
    # Verify model has forward method
    if not hasattr(model, 'forward') or model.forward.__name__ == '_forward_unimplemented':
        raise RuntimeError(
            "Model doesn't have a proper forward method. "
            "Check if the model class is correctly loaded."
        )
    
    logger.info(f"Model loaded: {type(model).__name__}")
    logger.info(f"Model config: {model.config}")
    
    # Enable gradient checkpointing if needed
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Setup LoRA if requested
    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Prepare dataset
    train_dataset = TextSFTDataset(
        data_path=data_args.data_dir,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length
    )
    
    # Test a single sample
    logger.info("Testing data pipeline...")
    test_sample = train_dataset[0]
    logger.info(f"Sample keys: {test_sample.keys()}")
    logger.info(f"Input length: {len(test_sample['input_ids'])}")
    
    # Determine model type for data collator
    model_type = "qwen3omni" if "Qwen3OmniMoe" in type(model).__name__ else "standard"
    
    # Setup data collator
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        model_type=model_type
    )
    
    # Test collator with a small batch
    test_batch = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    try:
        collated = data_collator(test_batch)
        logger.info(f"Collated batch keys: {collated.keys()}")
        logger.info(f"Input shape: {collated['input_ids'].shape}")
        logger.info("✅ Data pipeline test passed!")
    except Exception as e:
        logger.error(f"❌ Collator test failed: {e}")
        raise

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    logger.info(f"Starting SFT for {type(model).__name__} on {len(train_dataset)} samples.")
    
    # Start training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    logger.info("Training finished. Saving final model and tokenizer.")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()