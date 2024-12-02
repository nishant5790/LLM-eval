from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset
import logging

class RagasEvaluator:
    def __init__(self):
        """Initialize RAGAS evaluator with default metrics"""
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_relevancy': context_relevancy,
            'context_recall': context_recall,
            'context_precision': context_precision
        }

    def evaluate_rag(self, 
                    questions, 
                    answers, 
                    contexts, 
                    metrics=None,
                    ground_truths=None):
        """
        Evaluate RAG (Retrieval Augmented Generation) outputs using RAGAS metrics
        
        :param questions: List of questions
        :param answers: List of generated answers
        :param contexts: List of contexts used for generation (list of lists)
        :param metrics: List of metric names to use (default: all available metrics)
        :param ground_truths: Optional list of ground truth answers
        :return: Dictionary containing evaluation results
        """
        try:
            # Prepare the dataset
            data = {
                'question': questions,
                'answer': answers,
                'contexts': contexts,
            }
            if ground_truths:
                data['ground_truth'] = ground_truths

            # Convert to RAGAS dataset format
            dataset = Dataset.from_dict(data)

            # Select metrics to use
            if metrics is None:
                metrics = list(self.metrics.values())
            else:
                metrics = [self.metrics[m] for m in metrics if m in self.metrics]

            # Run evaluation
            results = evaluate(
                dataset=dataset,
                metrics=metrics
            )

            return {
                'scores': results.to_dict(),
                'dataset_size': len(questions)
            }

        except Exception as e:
            self.logger.error(f"Error in RAGAS evaluation: {e}")
            return {'error': str(e)}

    def get_available_metrics(self):
        """Return list of available RAGAS metrics"""
        return list(self.metrics.keys())
