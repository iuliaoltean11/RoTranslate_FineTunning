import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
import numpy as np
from pathlib import Path
import json
import wandb
from typing import Dict, List
import evaluate


class TranslationTrainer:
    def __init__(self, model_name: str = "facebook/nllb-200-1.3B"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Load evaluation metrics
        self.bleu_metric = evaluate.load("sacrebleu")
        self.rouge_metric = evaluate.load("rouge")

        # LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=16,  # Low rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )

        print(" Model and tokenizer loaded successfully")

    def prepare_model_for_training(self):
        """Apply LoRA and prepare model for training"""
        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        # Enable training mode
        self.model.train()

        print(" Model prepared for LoRA training")

    def load_dataset(self, dataset_path: str):
        """Load processed dataset"""
        self.dataset = load_from_disk(dataset_path)
        print(f" Dataset loaded: {len(self.dataset['train'])} train samples")

        # Data collator
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean up texts
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        # BLEU score
        bleu_result = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )

        # ROUGE score
        rouge_result = self.rouge_metric.compute(
            predictions=decoded_preds,
            references=[ref[0] for ref in decoded_labels]
        )

        return {
            "bleu": bleu_result["score"],
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeL": rouge_result["rougeL"],
            "prediction_length": np.mean([len(pred.split()) for pred in decoded_preds])
        }

    def setup_training_args(self, output_dir: str) -> Seq2SeqTrainingArguments:
        """Setup training arguments"""
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,

            # Training parameters
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,

            # Optimization
            learning_rate=5e-4,
            weight_decay=0.01,
            warmup_steps=100,

            # Evaluation
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,

            # Generation parameters
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=4,

            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            report_to="wandb",

            # Performance
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False,

            # Early stopping
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,

            # Save settings
            save_total_limit=2,
            save_only_model=True,
        )

    def train(self, output_dir: str = "models/rotranslate"):
        """Train the model"""

        # Prepare model
        self.prepare_model_for_training()

        # Setup training arguments
        training_args = self.setup_training_args(output_dir)

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print(" Starting training...")

        # Train
        trainer.train()

        # Save final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f" Training completed! Model saved to {output_dir}")

        # Evaluate on test set
        print(" Evaluating on test set...")
        test_results = trainer.evaluate(self.dataset["test"])

        # Save results
        with open(f"{output_dir}/test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

        print(f"Test BLEU: {test_results.get('eval_bleu', 'N/A'):.4f}")
        print(f"Test ROUGE-L: {test_results.get('eval_rougeL', 'N/A'):.4f}")

        return trainer


def main():
    # Initialize wandb
    wandb.init(
        project="rotranslate-finetuning",
        config={
            "model": "facebook/nllb-200-1.3B",
            "task": "ro-en-translation",
            "method": "lora-finetuning"
        }
    )

    # Initialize trainer
    trainer = TranslationTrainer()

    # Load dataset
    trainer.load_dataset("data/processed/translation_dataset")

    # Train model
    trained_model = trainer.train("models/rotranslate")

    print(" Training pipeline completed successfully!")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()