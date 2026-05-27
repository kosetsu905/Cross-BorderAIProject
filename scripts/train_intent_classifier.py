from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


INTENT_LABELS = {
    "pre_sales": 0,
    "order_fulfillment": 1,
    "post_sales_support": 2,
}
ID_TO_LABEL = {value: key for key, value in INTENT_LABELS.items()}
ML_DEPENDENCY_HINT = (
    "Install optional ML dependencies first: "
    "python -m pip install -r requirements-ml.txt"
)


@dataclass
class TrainingConfig:
    model_name: str = "bert-base-multilingual-cased"
    max_length: int = 128
    num_labels: int = 3
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    output_dir: str = "artifacts/intent_classifier_v1"
    languages: list[str] = field(
        default_factory=lambda: ["en", "es", "fr", "de", "ja", "zh", "ar", "pt", "ko", "it"]
    )


class IntentDatasetBuilder:
    INTENT_LABELS = INTENT_LABELS
    SEED_SAMPLES: dict[str, dict[str, list[str]]] = {
        "en": {
            "pre_sales": [
                "Does this camera work with Alexa?",
                "What is the difference between Basic and Pro model?",
                "Is this compatible with 220V power?",
            ],
            "order_fulfillment": [
                "Where is my order #12345?",
                "Can I change the shipping address?",
                "When will my package arrive?",
            ],
            "post_sales_support": [
                "The item arrived damaged, how do I return it?",
                "How do I set up this product?",
                "I need a replacement part",
            ],
        },
        "es": {
            "pre_sales": [
                "Esta cámara funciona con Alexa?",
                "Cuál es la diferencia entre Basic y Pro?",
                "Es compatible con corriente de 220V?",
            ],
            "order_fulfillment": [
                "Dónde está mi pedido #12345?",
                "Puedo cambiar la dirección de envío?",
                "Cuándo llegará mi paquete?",
            ],
            "post_sales_support": [
                "El producto llegó dañado, cómo lo devuelvo?",
                "Cómo configuro este producto?",
                "Necesito una pieza de reemplazo",
            ],
        },
        "fr": {
            "pre_sales": [
                "Cette caméra fonctionne-t-elle avec Alexa?",
                "Quelle est la différence entre Basic et Pro?",
                "Est-ce compatible avec une alimentation 220V?",
            ],
            "order_fulfillment": [
                "Où est ma commande #12345?",
                "Puis-je changer l'adresse de livraison?",
                "Quand mon colis arrivera-t-il?",
            ],
            "post_sales_support": [
                "L'article est arrivé endommagé, comment le retourner?",
                "Comment configurer ce produit?",
                "J'ai besoin d'une pièce de rechange",
            ],
        },
        "de": {
            "pre_sales": [
                "Funktioniert diese Kamera mit Alexa?",
                "Was ist der Unterschied zwischen Basic und Pro?",
                "Ist sie mit 220V Strom kompatibel?",
            ],
            "order_fulfillment": [
                "Wo ist meine Bestellung #12345?",
                "Kann ich die Lieferadresse ändern?",
                "Wann kommt mein Paket an?",
            ],
            "post_sales_support": [
                "Der Artikel kam beschädigt an, wie sende ich ihn zurück?",
                "Wie richte ich dieses Produkt ein?",
                "Ich brauche ein Ersatzteil",
            ],
        },
        "ja": {
            "pre_sales": [
                "このカメラはアレクサと連携できますか?",
                "Basic と Pro の違いは何ですか?",
                "220V 電源に対応していますか?",
            ],
            "order_fulfillment": [
                "注文 #12345 はどこにありますか?",
                "配送先住所を変更できますか?",
                "いつ届きますか?",
            ],
            "post_sales_support": [
                "商品が破損して届きました、返品方法を教えてください",
                "この製品のセットアップ方法を教えてください",
                "交換部品が必要です",
            ],
        },
        "zh": {
            "pre_sales": [
                "这个摄像头支持 Alexa 吗?",
                "Basic 和 Pro 有什么区别?",
                "这个产品支持 220V 电源吗?",
            ],
            "order_fulfillment": [
                "我的订单 #12345 到哪里了?",
                "我可以修改收货地址吗?",
                "包裹什么时候送达?",
            ],
            "post_sales_support": [
                "商品收到时已经损坏，怎么退货?",
                "这个产品怎么安装?",
                "我需要更换配件",
            ],
        },
        "ar": {
            "pre_sales": [
                "هل تعمل هذه الكاميرا مع أليكسا؟",
                "ما الفرق بين طراز Basic و Pro؟",
                "هل هذا المنتج متوافق مع كهرباء 220 فولت؟",
            ],
            "order_fulfillment": [
                "أين طلبي رقم 12345؟",
                "هل يمكنني تغيير عنوان الشحن؟",
                "متى سيصل الطرد؟",
            ],
            "post_sales_support": [
                "وصل المنتج تالفا، كيف أعيده؟",
                "كيف أقوم بإعداد هذا المنتج؟",
                "أحتاج إلى قطعة بديلة",
            ],
        },
        "pt": {
            "pre_sales": [
                "Esta câmera funciona com Alexa?",
                "Qual é a diferença entre Basic e Pro?",
                "É compatível com energia 220V?",
            ],
            "order_fulfillment": [
                "Onde está meu pedido #12345?",
                "Posso alterar o endereço de entrega?",
                "Quando meu pacote vai chegar?",
            ],
            "post_sales_support": [
                "O item chegou danificado, como faço a devolução?",
                "Como configuro este produto?",
                "Preciso de uma peça de reposição",
            ],
        },
        "ko": {
            "pre_sales": [
                "이 카메라는 Alexa와 작동하나요?",
                "Basic과 Pro 모델의 차이는 무엇인가요?",
                "220V 전원을 지원하나요?",
            ],
            "order_fulfillment": [
                "주문 #12345는 어디에 있나요?",
                "배송 주소를 변경할 수 있나요?",
                "패키지는 언제 도착하나요?",
            ],
            "post_sales_support": [
                "상품이 파손되어 도착했습니다. 어떻게 반품하나요?",
                "이 제품은 어떻게 설정하나요?",
                "교체 부품이 필요합니다",
            ],
        },
        "it": {
            "pre_sales": [
                "Questa fotocamera funziona con Alexa?",
                "Qual è la differenza tra Basic e Pro?",
                "È compatibile con alimentazione 220V?",
            ],
            "order_fulfillment": [
                "Dov'è il mio ordine #12345?",
                "Posso cambiare l'indirizzo di spedizione?",
                "Quando arriverà il pacco?",
            ],
            "post_sales_support": [
                "L'articolo è arrivato danneggiato, come posso restituirlo?",
                "Come configuro questo prodotto?",
                "Ho bisogno di un pezzo di ricambio",
            ],
        },
    }

    @classmethod
    def build_samples(cls, config: TrainingConfig, include_augmentation: bool = True) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for language in config.languages:
            language_samples = cls.SEED_SAMPLES.get(language, cls.SEED_SAMPLES["en"])
            for intent, examples in language_samples.items():
                for text in examples:
                    samples.append(
                        {
                            "text": text,
                            "label": cls.INTENT_LABELS[intent],
                            "language": language,
                            "intent": intent,
                            "augmented": False,
                        }
                    )
        if include_augmentation:
            samples.extend(cls._augment_with_mock_backtranslation(samples))
        return samples

    @classmethod
    def build_dataset(cls, config: TrainingConfig):
        try:
            from datasets import Dataset, DatasetDict
        except ImportError as exc:
            raise RuntimeError(ML_DEPENDENCY_HINT) from exc

        dataset = Dataset.from_list(cls.build_samples(config))
        split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
        return DatasetDict({"train": split["train"], "validation": split["test"]})

    @classmethod
    def dry_run_summary(cls, config: TrainingConfig) -> dict[str, Any]:
        samples = cls.build_samples(config)
        by_language = {language: 0 for language in config.languages}
        by_intent = {intent: 0 for intent in cls.INTENT_LABELS}
        for sample in samples:
            by_language[sample["language"]] = by_language.get(sample["language"], 0) + 1
            by_intent[sample["intent"]] = by_intent.get(sample["intent"], 0) + 1
        return {
            "model_name": config.model_name,
            "output_dir": config.output_dir,
            "num_labels": config.num_labels,
            "languages": config.languages,
            "sample_count": len(samples),
            "by_language": by_language,
            "by_intent": by_intent,
            "label2id": cls.INTENT_LABELS,
        }

    @staticmethod
    def _augment_with_mock_backtranslation(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        augmented: list[dict[str, Any]] = []
        for sample in samples:
            if sample["language"] == "en":
                augmented.append({**sample, "text": sample["text"].replace("?", " ?").strip(), "augmented": True})
        return augmented


class IntentClassifierTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(ML_DEPENDENCY_HINT) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = None

    def tokenize_function(self, examples: dict[str, Any]) -> dict[str, Any]:
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
        )

    def compute_metrics(self, eval_pred) -> dict[str, float]:
        try:
            import torch
            from sklearn.metrics import accuracy_score, f1_score
        except ImportError as exc:
            raise RuntimeError(ML_DEPENDENCY_HINT) from exc

        predictions, labels = eval_pred
        preds = torch.argmax(torch.tensor(predictions), dim=1).numpy()
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
            "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        }

    def train(self, dataset) -> str:
        try:
            from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback, Trainer, TrainingArguments
        except ImportError as exc:
            raise RuntimeError(ML_DEPENDENCY_HINT) from exc

        tokenized = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text", "language", "intent", "augmented"],
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            id2label=ID_TO_LABEL,
            label2id=INTENT_LABELS,
        )
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            push_to_hub=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        logger.info("Starting multilingual intent classifier training")
        trainer.train()

        model_path = Path(self.config.output_dir) / "final"
        trainer.save_model(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
        export_label_map(model_path)
        logger.info("Model saved to %s", model_path)
        return str(model_path)


class IntentClassifierTool:
    def __init__(self, model_path: str | Path, classifier: Any | None = None):
        self.model_path = Path(model_path)
        self.id2label = load_label_map(self.model_path)
        if classifier is not None:
            self.classifier = classifier
            return
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(ML_DEPENDENCY_HINT) from exc
        self.classifier = pipeline(
            "text-classification",
            model=str(self.model_path),
            tokenizer=str(self.model_path),
            framework="pt",
            device=0 if torch.cuda.is_available() else -1,
        )

    def predict(self, text: str, language: str = "en") -> dict[str, Any]:
        raw_result = self.classifier(str(text or "")[:512])[0]
        intent = self._label_to_intent(raw_result.get("label"))
        confidence = float(raw_result.get("score") or 0)
        return {
            "detected_intent": intent,
            "confidence_score": confidence,
            "requires_human_review": confidence < 0.75,
            "language_detected": language,
        }

    def _label_to_intent(self, label: Any) -> str:
        if label in INTENT_LABELS:
            return str(label)
        label_text = str(label)
        if label_text.startswith("LABEL_"):
            label_text = label_text.replace("LABEL_", "", 1)
        return self.id2label.get(label_text, "post_sales_support")


def export_label_map(model_path: str | Path) -> None:
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "label_map.json").open("w", encoding="utf-8") as file:
        json.dump({"id2label": {str(key): value for key, value in ID_TO_LABEL.items()}, "label2id": INTENT_LABELS}, file, indent=2)


def load_label_map(model_path: str | Path) -> dict[str, str]:
    with (Path(model_path) / "label_map.json").open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return {str(key): str(value) for key, value in payload["id2label"].items()}


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        languages=[item.strip() for item in args.languages.split(",") if item.strip()],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or dry-run the multilingual Customer Service intent classifier.")
    parser.add_argument("--dry-run", action="store_true", help="Print dataset summary without importing ML dependencies.")
    parser.add_argument("--train", action="store_true", help="Train multilingual-BERT and save the final model.")
    parser.add_argument("--model-name", default="bert-base-multilingual-cased")
    parser.add_argument("--output-dir", default="artifacts/intent_classifier_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--languages", default="en,es,fr,de,ja,zh,ar,pt,ko,it")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    config = _config_from_args(args)
    if args.train:
        dataset = IntentDatasetBuilder.build_dataset(config)
        trainer = IntentClassifierTrainer(config)
        model_path = trainer.train(dataset)
        print(json.dumps({"status": "trained", "model_path": model_path}, indent=2))
        return 0

    summary = IntentDatasetBuilder.dry_run_summary(config)
    print(json.dumps({"status": "dry_run", "config": asdict(config), "dataset": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
