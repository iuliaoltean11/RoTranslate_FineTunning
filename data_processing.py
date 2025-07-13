import json
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict
import numpy as np
from transformers import AutoTokenizer


class TranslationDataProcessor:
    def __init__(self, model_name: str = "facebook/nllb-200-1.3B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 512

    def load_raw_data(self, file_path: str) -> List[Dict]:
        """Load raw JSON data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep Romanian diacritics
        text = re.sub(r'[^\w\s\.,!?;:\-\(\)\'\"ăâîșțĂÂÎȘȚ]', '', text)

        # Normalize punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)

        return text.strip()

    def filter_by_length(self, data: List[Dict]) -> List[Dict]:
        """Filter out samples that are too short or too long"""
        filtered = []

        for sample in data:
            source_len = len(sample['source'].split())
            target_len = len(sample['target'].split())

            # Filter criteria
            if (5 <= source_len <= 100 and
                    5 <= target_len <= 100 and
                    abs(source_len - target_len) <= 50):  # Length ratio check
                filtered.append(sample)

        print(f" Filtered from {len(data)} to {len(filtered)} samples")
        return filtered

    def detect_language_quality(self, text: str, expected_lang: str) -> bool:
        """Simple language detection for quality control"""

        # Romanian specific patterns
        ro_patterns = [
            r'[ăâîșț]',  # Romanian diacritics
            r'\b(și|sau|dar|pentru|această|această|este|sunt|cu|de|la|în)\b',
            r'\b(dezvoltare|sistem|algoritm|programare|aplicație)\b'
        ]

        # English specific patterns
        en_patterns = [
            r'\b(the|and|or|but|for|this|that|is|are|with|of|to|in)\b',
            r'\b(development|system|algorithm|programming|application)\b'
        ]

        if expected_lang == 'ro':
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in ro_patterns)
        else:
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in en_patterns)

    def quality_filter(self, data: List[Dict]) -> List[Dict]:
        """Filter samples based on quality criteria"""
        high_quality = []

        for sample in data:
            source = sample['source']
            target = sample['target']

            # Basic quality checks
            if (self.detect_language_quality(source, 'ro') and
                    self.detect_language_quality(target, 'en') and
                    len(source.strip()) > 10 and
                    len(target.strip()) > 10 and
                    not source.lower() == target.lower()):  # Not identical

                high_quality.append(sample)

        print(f" Quality filtered from {len(data)} to {len(high_quality)} samples")
        return high_quality

    def create_training_format(self, data: List[Dict]) -> Dict:
        """Convert to training format with proper tokenization"""

        # Prepare data for NLLB format
        processed_data = {
            'translation': []
        }

        for sample in data:
            # Clean texts
            source_text = self.clean_text(sample['source'])
            target_text = self.clean_text(sample['target'])

            # NLLB format: source and target with language codes
            processed_data['translation'].append({
                'ron_Latn': source_text,  # Romanian
                'eng_Latn': target_text  # English
            })

        return processed_data

    def tokenize_function(self, examples):
        """Tokenization function for training"""
        # Get source and target texts
        sources = [ex['ron_Latn'] for ex in examples['translation']]
        targets = [ex['eng_Latn'] for ex in examples['translation']]

        # Tokenize with language codes
        model_inputs = self.tokenizer(
            sources,
            text_target=targets,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        return model_inputs

    def create_dataset_splits(self, data: List[Dict]) -> DatasetDict:
        """Create train/validation/test splits"""

        # Shuffle data
        np.random.shuffle(data)

        # Split ratios
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))

        # Create splits
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        # Convert to HuggingFace format
        train_formatted = self.create_training_format(train_data)
        val_formatted = self.create_training_format(val_data)
        test_formatted = self.create_training_format(test_data)

        # Create datasets
        dataset_dict = DatasetDict({
            'train': Dataset.from_dict(train_formatted),
            'validation': Dataset.from_dict(val_formatted),
            'test': Dataset.from_dict(test_formatted)
        })

        print(f" Dataset splits created:")
        print(f"  Train: {len(dataset_dict['train'])}")
        print(f"  Validation: {len(dataset_dict['validation'])}")
        print(f"  Test: {len(dataset_dict['test'])}")

        return dataset_dict

    def save_processed_data(self, dataset_dict: DatasetDict, output_dir: str):
        """Save processed dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save dataset
        dataset_dict.save_to_disk(output_path)

        # Save statistics
        stats = {
            'total_samples': len(dataset_dict['train']) + len(dataset_dict['validation']) + len(dataset_dict['test']),
            'train_samples': len(dataset_dict['train']),
            'validation_samples': len(dataset_dict['validation']),
            'test_samples': len(dataset_dict['test']),
            'max_length': self.max_length,
            'tokenizer': self.tokenizer.name_or_path
        }

        with open(output_path / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f" Processed dataset saved to {output_path}")


def main():
    processor = TranslationDataProcessor()

    print(" Starting data preprocessing...")

    # Load raw data
    raw_data = processor.load_raw_data("data/raw/combined_dataset.json")
    print(f" Loaded {len(raw_data)} raw samples")

    # Apply filters
    filtered_data = processor.filter_by_length(raw_data)
    quality_data = processor.quality_filter(filtered_data)

    # Create dataset splits
    dataset_dict = processor.create_dataset_splits(quality_data)

    # Tokenize datasets
    tokenized_datasets = dataset_dict.map(
        processor.tokenize_function,
        batched=True,
        remove_columns=['translation']
    )

    # Save processed data
    processor.save_processed_data(tokenized_datasets, "data/processed/translation_dataset")

    print(" Data preprocessing complete!")

    # Print sample
    print("\n Sample processed data:")
    sample = dataset_dict['train'][0]
    print(f"Source: {sample['translation']['ron_Latn']}")
    print(f"Target: {sample['translation']['eng_Latn']}")


if __name__ == "__main__":
    main()