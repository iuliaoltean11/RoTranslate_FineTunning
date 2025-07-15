# run_pipeline.py
"""
Complete pipeline for RoTranslate project
Runs data collection, preprocessing, training, and evaluation
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import json


class RoTranslatePipeline:
    def __init__(self):
        self.project_root = Path.cwd()
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "data/raw",
            "data/processed",
            "models",
            "results",
            "logs",
            "src/data_collection",
            "src/preprocessing",
            "src/training",
            "src/inference",
            "configs"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

        print(" Project directories created")

    def check_requirements(self):
        """Check if required packages are installed"""
        try:
            import torch
            import transformers
            import datasets
            import peft
            import evaluate
            print(" All required packages are installed")
            return True
        except ImportError as e:
            print(f" Missing package: {e}")
            print("Please run: pip install -r requirements.txt")
            return False

    def run_data_collection(self):
        """Run data collection script"""
        print("\n Starting data collection...")

        script_path = "src/data_collection/scraper.py"
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

        if result.returncode == 0:
            print(" Data collection completed successfully")
            return True
        else:
            print(f" Data collection failed: {result.stderr}")
            return False

    def run_preprocessing(self):
        """Run data preprocessing"""
        print("\n Starting data preprocessing...")

        script_path = "src/preprocessing/data_processor.py"
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

        if result.returncode == 0:
            print(" Data preprocessing completed successfully")
            return True
        else:
            print(f" Data preprocessing failed: {result.stderr}")
            return False

    def run_training(self, epochs: int = 3):
        """Run model training"""
        print(f"\n Starting model training ({epochs} epochs)...")

        # Set environment variables for training
        env = os.environ.copy()
        env['WANDB_PROJECT'] = 'rotranslate-finetuning'

        script_path = "src/training/train_model.py"
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode == 0:
            print(" Model training completed successfully")
            return True
        else:
            print(f" Model training failed: {result.stderr}")
            return False

    def run_evaluation(self):
        """Run model evaluation"""
        print("\n Starting model evaluation...")

        script_path = "src/inference/translator.py"
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

        if result.returncode == 0:
            print(" Model evaluation completed successfully")
            return True
        else:
            print(f" Model evaluation failed: {result.stderr}")
            return False

    def create_config_file(self):
        """Create configuration file"""
        config = {
            "model": {
                "base_model": "facebook/nllb-200-1.3B",
                "max_length": 512,
                "target_languages": ["ron_Latn", "eng_Latn"]
            },
            "training": {
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 5e-4,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1
            },
            "data": {
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "min_length": 5,
                "max_length": 100
            },
            "evaluation": {
                "metrics": ["bleu", "rouge"],
                "num_beams": 4
            }
        }

        with open("configs/config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(" Configuration file created")

    def generate_readme(self):
        """Generate README file"""
        readme_content = """# RoTranslate - Fine-tuned Romanian-English Translation Model

## Overview
RoTranslate is a specialized translation model fine-tuned for technical Romanian-English translation using LoRA (Low-Rank Adaptation) on the NLLB-200 model.

## Features
- 🇷🇴 Romanian to English translation
-  Specialized for technical terminology
-  Fast inference with LoRA
-  Comprehensive evaluation metrics
-  Docker support for deployment

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python run_pipeline.py --full-pipeline
```

### 3. Individual Steps
```bash
# Data collection
python src/data_collection/scraper.py

# Preprocessing
python src/preprocessing/data_processor.py

# Training
python src/training/train_model.py

# Evaluation
python src/inference/translator.py
```

## Project Structure
```
rotranslate/
├── data/               # Datasets
├── src/                # Source code
├── models/             # Trained models
├── results/            # Evaluation results
├── configs/            # Configuration files
└── requirements.txt
```

## Performance
- BLEU Score: ~XX.X (on technical test set)
- ROUGE-L: ~XX.X
- Training Time: ~X hours on GPU

## Usage Example
```python
from src.inference.translator import RoTranslator

translator = RoTranslator("models/rotranslate")
result = translator.translate("Algoritmul de învățare automată procesează datele.")
print(result)  # "The machine learning algorithm processes the data."
```

## Technical Details
- **Base Model**: NLLB-200-1.3B
- **Fine-tuning**: LoRA with r=16
- **Training Data**: Technical documents + glossaries
- **Languages**: Romanian → English
- **Specialization**: IT, AI, Software Development

## Evaluation Results
See `results/evaluation_results.json` for detailed metrics.

## Contributing
Feel free to contribute by adding more technical domain data or improving the model architecture.

## License
MIT License
"""

        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        print( README.md generated")

    def run_full_pipeline(self, epochs: int = 3):
        """Run the complete pipeline"""
        print(" Starting RoTranslate Full Pipeline")
        print("=" * 50)

        # Check requirements
        if not self.check_requirements():
            return False

        # Create config
        self.create_config_file()

        # Run pipeline steps
        steps = [
            ("Data Collection", self.run_data_collection),
            ("Data Preprocessing", self.run_preprocessing),
            ("Model Training", lambda: self.run_training(epochs)),
            ("Model Evaluation", self.run_evaluation)
        ]

        for step_name, step_func in steps:
            print(f"\n{'=' * 20} {step_name} {'=' * 20}")
            if not step_func():
                print(f" Pipeline failed at: {step_name}")
                return False

        # Generate documentation
        self.generate_readme()

        print("\n" + "=" * 50)
        print(" RoTranslate pipeline completed successfully!")
        print(" Check the following directories:")
        print("  - models/rotranslate/ (trained model)")
        print("  - results/ (evaluation results)")
        print("  - data/ (processed datasets)")

        return True


def main():
    parser = argparse.ArgumentParser(description="RoTranslate Pipeline Runner")
    parser.add_argument("--full-pipeline", action="store_true", help="Run complete pipeline")
    parser.add_argument("--data-only", action="store_true", help="Run only data collection and preprocessing")
    parser.add_argument("--train-only", action="store_true", help="Run only training")
    parser.add_argument("--eval-only", action="store_true", help="Run only evaluation")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")

    args = parser.parse_args()

    pipeline = RoTranslatePipeline()

    if args.full_pipeline:
        pipeline.run_full_pipeline(args.epochs)
    elif args.data_only:
        pipeline.run_data_collection()
        pipeline.run_preprocessing()
    elif args.train_only:
        pipeline.run_training(args.epochs)
    elif args.eval_only:
        pipeline.run_evaluation()
    else:
        print("Please specify an option. Use --help for more information.")
        print("\nQuick start: python run_pipeline.py --full-pipeline")


if __name__ == "__main__":
    main()