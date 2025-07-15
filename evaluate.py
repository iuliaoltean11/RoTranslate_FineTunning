import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import json
from pathlib import Path
from typing import List, Dict
import evaluate
import time


class RoTranslator:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-1.3B",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()

        print(f" Model loaded from {model_path}")

    def translate(self, text: str, max_length: int = 128) -> str:
        """Translate Romanian text to English"""

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation.strip()

    def translate_batch(self, texts: List[str], max_length: int = 128) -> List[str]:
        """Translate multiple texts at once"""

        # Tokenize all inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate translations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode all outputs
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [t.strip() for t in translations]


class ModelEvaluator:
    def __init__(self, model_path: str):
        self.translator = RoTranslator(model_path)
        self.bleu_metric = evaluate.load("sacrebleu")
        self.rouge_metric = evaluate.load("rouge")

    def evaluate_on_test_set(self, test_data_path: str) -> Dict:
        """Evaluate model on test dataset"""

        # Load test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        print(f"📊 Evaluating on {len(test_data)} test samples...")

        # Prepare data
        source_texts = [item['source'] for item in test_data]
        reference_texts = [item['target'] for item in test_data]

        # Translate in batches
        batch_size = 8
        predictions = []

        for i in range(0, len(source_texts), batch_size):
            batch = source_texts[i:i + batch_size]
            batch_predictions = self.translator.translate_batch(batch)
            predictions.extend(batch_predictions)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(source_texts)} samples")

        # Calculate metrics
        bleu_score = self.bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in reference_texts]
        )

        rouge_scores = self.rouge_metric.compute(
            predictions=predictions,
            references=reference_texts
        )

        results = {
            'bleu': bleu_score['score'],
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'num_samples': len(test_data),
            'avg_prediction_length': sum(len(p.split()) for p in predictions) / len(predictions),
            'avg_reference_length': sum(len(r.split()) for r in reference_texts) / len(reference_texts)
        }

        return results, predictions, reference_texts, source_texts

    def evaluate_technical_terms(self) -> Dict:
        """Evaluate on technical terminology"""

        technical_test_cases = [
            {
                'source': 'Algoritmul de învățare automată procesează datele de intrare.',
                'target': 'The machine learning algorithm processes the input data.',
                'category': 'AI/ML'
            },
            {
                'source': 'Baza de date relațională stochează informațiile în tabele.',
                'target': 'The relational database stores information in tables.',
                'category': 'Database'
            },
            {
                'source': 'Interfața utilizator este intuitivă și ușor de folosit.',
                'target': 'The user interface is intuitive and easy to use.',
                'category': 'UI/UX'
            },
            {
                'source': 'Sistemul de securitate cibernetică protejează împotriva atacurilor.',
                'target': 'The cybersecurity system protects against attacks.',
                'category': 'Security'
            },
            {
                'source': 'Aplicația web folosește arhitectura microserviciilor.',
                'target': 'The web application uses microservices architecture.',
                'category': 'Architecture'
            },
            {
                'source': 'Rețeaua neurală convoluțională recunoaște imaginile.',
                'target': 'The convolutional neural network recognizes images.',
                'category': 'Deep Learning'
            },
            {
                'source': 'API-ul REST permite comunicarea între servicii.',
                'target': 'The REST API enables communication between services.',
                'category': 'API'
            },
            {
                'source': 'Blockchain-ul asigură transparența tranzacțiilor.',
                'target': 'Blockchain ensures transaction transparency.',
                'category': 'Blockchain'
            }
        ]

        print(" Evaluating technical terminology...")

        # Translate technical terms
        source_texts = [case['source'] for case in technical_test_cases]
        predictions = self.translator.translate_batch(source_texts)

        # Calculate scores by category
        category_results = {}

        for i, case in enumerate(technical_test_cases):
            category = case['category']
            if category not in category_results:
                category_results[category] = {'predictions': [], 'references': []}

            category_results[category]['predictions'].append(predictions[i])
            category_results[category]['references'].append(case['target'])

        # Calculate metrics per category
        detailed_results = {}
        overall_predictions = []
        overall_references = []

        for category, data in category_results.items():
            preds = data['predictions']
            refs = data['references']

            bleu = self.bleu_metric.compute(
                predictions=preds,
                references=[[r] for r in refs]
            )

            detailed_results[category] = {
                'bleu': bleu['score'],
                'samples': len(preds)
            }

            overall_predictions.extend(preds)
            overall_references.extend(refs)

        # Overall technical terms performance
        overall_bleu = self.bleu_metric.compute(
            predictions=overall_predictions,
            references=[[r] for r in overall_references]
        )

        detailed_results['overall'] = {
            'bleu': overall_bleu['score'],
            'total_samples': len(overall_predictions)
        }

        return detailed_results, predictions, technical_test_cases


def demo_translation():
    """Demo function to show translation capabilities"""

    model_path = "models/rotranslate"
    translator = RoTranslator(model_path)

    demo_texts = [
        "Dezvoltarea aplicațiilor web moderne necesită cunoștințe avansate de programare.",
        "Algoritmul de inteligență artificială analizează datele în timp real.",
        "Securitatea cibernetică este crucială pentru protejarea informațiilor sensibile.",
        "Baza de date NoSQL oferă flexibilitate în stocarea documentelor.",
        "Microserviciile permit scalabilitatea independentă a componentelor aplicației."
    ]

    print(" RoTranslate Demo")
    print("=" * 60)

    for i, text in enumerate(demo_texts, 1):
        print(f"\n{i}. Romanian: {text}")

        start_time = time.time()
        translation = translator.translate(text)
        end_time = time.time()

        print(f"   English: {translation}")
        print(f"   Time: {end_time - start_time:.2f}s")


def main():
    """Main evaluation script"""

    model_path = "models/rotranslate"
    evaluator = ModelEvaluator(model_path)

    print(" Starting model evaluation...")

    # 1. Demo translation
    demo_translation()

    # 2. Evaluate on technical terms
    print("\n" + "=" * 60)
    tech_results, tech_predictions, tech_cases = evaluator.evaluate_technical_terms()

    print(" Technical Terms Evaluation Results:")
    for category, results in tech_results.items():
        print(f"  {category}: BLEU = {results['bleu']:.2f}")

    # 3. Show some technical examples
    print("\n Technical Translation Examples:")
    for i, (pred, case) in enumerate(zip(tech_predictions[:3], tech_cases[:3])):
        print(f"\n{i + 1}. Source: {case['source']}")
        print(f"   Predicted: {pred}")
        print(f"   Reference: {case['target']}")
        print(f"   Category: {case['category']}")

    # 4. If test data exists, evaluate on full test set
    test_data_path = "data/raw/test_set.json"
    if Path(test_data_path).exists():
        print("\n" + "=" * 60)
        print(" Full Test Set Evaluation:")

        results, predictions, references, sources = evaluator.evaluate_on_test_set(test_data_path)

        print(f"BLEU Score: {results['bleu']:.2f}")
        print(f"ROUGE-1: {results['rouge1']:.2f}")
        print(f"ROUGE-2: {results['rouge2']:.2f}")
        print(f"ROUGE-L: {results['rougeL']:.2f}")

        # Save detailed results
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "technical_terms": tech_results,
                "full_test_set": results
            }, f, ensure_ascii=False, indent=2)

        print(f" Detailed results saved to {output_dir}/evaluation_results.json")

    print("\n Evaluation completed!")


if __name__ == "__main__":
    main()