from datetime import datetime
from langsmith import Client
from langsmith.utils import LangSmithConflictError
from typing import Dict, List

class EvaluationDatasetManager:
    def __init__(self):
        self.client = Client()
        self.evaluation_examples = [
            {
                "question": "What are the key concepts of deep learning?",
                "expected_answer": "Deep learning involves neural networks processing data through multiple layers to learn hierarchical representations. The key concepts include backpropagation, activation functions, and gradient descent optimization.",
            },
            {
                "question": "How does transfer learning work?",
                "expected_answer": "Transfer learning allows models to apply knowledge learned from one task to another related task. It involves taking a pre-trained model and fine-tuning it on a new dataset or task while preserving learned features.",
            },
            {
                "question": "What is the role of attention mechanisms in transformers?",
                "expected_answer": "Attention mechanisms in transformers enable the model to focus on relevant parts of the input sequence when making predictions. They compute weighted relationships between all elements in the sequence, allowing for better handling of long-range dependencies.",
            },
            {
                "question": "Explain the concept of embeddings in machine learning.",
                "expected_answer": "Embeddings are dense vector representations that capture semantic relationships between items in a continuous space. They convert discrete data like words or categories into numerical vectors that preserve meaningful relationships.",
            },
            {
                "question": "What impact did DeepSeek model have on OpenAI",
                "expected_answer": "DeepSeek had a significant impact on OpenAI by challenging it with a competitive large language model, DeepSeek-R1. This model implemented run-time reasoning similar to OpenAI's o1 but displayed its reasoning steps, making it more transparent. DeepSeek-R1 outperformed OpenAI's o1 on several benchmarks.",
            },
        ]

    def get_or_create_dataset(self) -> Dict:
        """Get existing dataset or create a new one with a unique name"""
        base_name = "the-batch-rag-evaluation"
        dataset_name = base_name
        
        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Evaluation dataset for The Batch RAG application"
            )
        except LangSmithConflictError:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"{base_name}_{timestamp}"
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description=f"Evaluation dataset for The Batch RAG application (Created: {timestamp})"
            )
        
        print(f"Using dataset: {dataset_name}")
        return dataset

    def create_examples(self, dataset_id: str) -> None:
        """Create examples in the evaluation dataset"""
        for example in self.evaluation_examples:
            self.client.create_example(
                inputs={"question": example["question"]},
                outputs={"expected_answer": example["expected_answer"]},
                dataset_id=dataset_id
            )

    def get_examples(self) -> List[Dict]:
        """Get the list of evaluation examples"""
        return self.evaluation_examples