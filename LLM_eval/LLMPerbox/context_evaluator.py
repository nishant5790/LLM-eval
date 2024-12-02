from typing import List, Dict, Any
import logging
import re

class ContextEvaluator:
    def __init__(self):
        """Initialize Context Evaluator with predefined prompts"""
        self.logger = logging.getLogger(__name__)
        
        # Predefined evaluation prompts
        self.evaluation_prompts = {
            'relevance': """
            Evaluate the relevance of the given context to the question and provide a relevancy score.
            
            Question: {question}
            Context: {context}
            
            Please analyze and rate each aspect on a scale of 1-10:
            1. Direct Relevance: How directly does the context address the question?
            2. Information Coverage: Does the context contain the necessary information?
            3. Conciseness: Is the context focused or contains unnecessary information?
            
            For each aspect, provide:
            - Score (1-10)
            - Brief explanation
            
            Finally, calculate the overall relevancy score as the average of the three scores.
            
            Format your response as:
            Direct Relevance: [score]
            [explanation]
            
            Information Coverage: [score]
            [explanation]
            
            Conciseness: [score]
            [explanation]
            
            Overall Relevancy Score: [average_score]
            """,
            
            'factual_consistency': """
            Analyze the factual consistency between the answer and the provided context.
            Rate each aspect on a scale of 1-10.
            
            Question: {question}
            Context: {context}
            Answer: {answer}
            
            Please evaluate:
            1. Factual Accuracy: Are all facts in the answer supported by the context?
            2. Completeness: Does the answer use all relevant facts from the context?
            3. Contradiction: Are there any contradictions? (10 = no contradictions)
            
            Format your response as:
            Factual Accuracy: [score]
            [explanation]
            
            Completeness: [score]
            [explanation]
            
            Contradiction Score: [score]
            [explanation]
            
            Overall Consistency Score: [average_score]
            """,
            
            'information_depth': """
            Assess the depth and quality of information in the context.
            Rate each aspect on a scale of 1-10.
            
            Question: {question}
            Context: {context}
            
            Evaluate:
            1. Detail Level: How detailed is the information provided?
            2. Comprehensiveness: Does it cover all aspects of the question?
            3. Technical Accuracy: Is the technical information accurate and well-explained?
            
            Format your response as:
            Detail Level: [score]
            [explanation]
            
            Comprehensiveness: [score]
            [explanation]
            
            Technical Accuracy: [score]
            [explanation]
            
            Overall Depth Score: [average_score]
            """,
            
            'coherence': """
            Evaluate the coherence and flow of information in the context.
            Rate each aspect on a scale of 1-10.
            
            Context: {context}
            
            Analyze:
            1. Logical Flow: How well does the information flow from one point to another?
            2. Structure: Is the information structured in a clear and organized manner?
            3. Clarity: How clear and understandable is the presentation of information?
            
            Format your response as:
            Logical Flow: [score]
            [explanation]
            
            Structure: [score]
            [explanation]
            
            Clarity: [score]
            [explanation]
            
            Overall Coherence Score: [average_score]
            """
        }

    def create_custom_prompt(self, prompt_template: str) -> None:
        """
        Add a custom evaluation prompt template
        
        :param prompt_template: Custom prompt template with placeholders
        :return: None
        """
        try:
            # Validate that the prompt contains at least one placeholder
            if not any(ph in prompt_template for ph in ['{question}', '{context}', '{answer}']):
                raise ValueError("Prompt template must contain at least one placeholder")
            
            prompt_name = f"custom_prompt_{len(self.evaluation_prompts)}"
            self.evaluation_prompts[prompt_name] = prompt_template
            return prompt_name
            
        except Exception as e:
            self.logger.error(f"Error creating custom prompt: {e}")
            return None

    def get_prompt(self, prompt_type: str) -> str:
        """
        Get a specific evaluation prompt
        
        :param prompt_type: Type of evaluation prompt
        :return: Prompt template string
        """
        return self.evaluation_prompts.get(prompt_type)

    def format_prompt(self, prompt_type: str, **kwargs) -> str:
        """
        Format a prompt with provided values
        
        :param prompt_type: Type of evaluation prompt
        :param kwargs: Values for prompt placeholders
        :return: Formatted prompt string
        """
        try:
            prompt_template = self.get_prompt(prompt_type)
            if not prompt_template:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
            
            return prompt_template.format(**kwargs)
            
        except Exception as e:
            self.logger.error(f"Error formatting prompt: {e}")
            return None

    def extract_scores(self, evaluation_text: str) -> Dict[str, Any]:
        """
        Extract numerical scores from evaluation text
        
        :param evaluation_text: Text containing evaluation scores
        :return: Dictionary of scores and overall score
        """
        try:
            # Extract individual scores
            scores = {}
            score_pattern = r'(\w+(?:\s+\w+)?): (\d+(?:\.\d+)?)'
            matches = re.finditer(score_pattern, evaluation_text)

            for match in matches:
                category = match.group(1)
                score = float(match.group(2))
                scores[category.lower().replace(' ', '_')] = score
            
            # Extract overall score
            overall_pattern = r'Overall.*Score: (\d+(?:\.\d+)?)'
            overall_match = re.search(overall_pattern, evaluation_text)
            if overall_match:
                scores['overall_score'] = float(overall_match.group(1))
            
            return {
                'scores': scores,
                'raw_evaluation': evaluation_text
            }
        
        except Exception as e:
            self.logger.error(f"Error extracting scores: {e}")
            return {
                'error': str(e),
                'raw_evaluation': evaluation_text
            }

    def evaluate_context(self, 
                        question: str, 
                        context: str, 
                        answer: str = None, 
                        prompt_types: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate context using specified prompt types
        
        :param question: Question being asked
        :param context: Context provided
        :param answer: Generated answer (optional)
        :param prompt_types: List of prompt types to use
        :return: Dictionary of formatted prompts for evaluation
        """
        if prompt_types is None:
            prompt_types = ['relevance', 'information_depth', 'coherence']
            if answer:
                prompt_types.append('factual_consistency')

        evaluation_prompts = {}
        try:
            for prompt_type in prompt_types:
                kwargs = {
                    'question': question,
                    'context': context
                }
                if answer and prompt_type == 'factual_consistency':
                    kwargs['answer'] = answer
                
                formatted_prompt = self.format_prompt(prompt_type, **kwargs)
                if formatted_prompt:
                    evaluation_prompts[prompt_type] = formatted_prompt

            return evaluation_prompts
            
        except Exception as e:
            self.logger.error(f"Error in context evaluation: {e}")
            return {'error': str(e)}

    def get_available_prompts(self) -> List[str]:
        """
        Get list of available evaluation prompt types
        
        :return: List of prompt type names
        """
        return list(self.evaluation_prompts.keys())
