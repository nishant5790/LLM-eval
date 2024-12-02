import os
import ast
def main():

    try:
        # Initialize the LLM Evaluator
        evaluator = LLMEvaluator()

        # Example generated text and reference text
        generated_text = "The quick brown fox jumps over the lazy dog."
        reference_text = "A quick brown fox jumps over a lazy dog."

        print("Running basic metric evaluations...")
        # Evaluate using multiple metrics
        results = evaluator.evaluate(
            generated_text, 
            reference_text, 
            metrics=['rouge', 'bleu', 'semantic_similarity', 'meteor_score'],
            bedrock_model=None
        )
        print("\nMetric Evaluation Results:")
        print(results)


        # Only run Bedrock evaluation if AWS credentials are configured
        try:
            print("\nAttempting Bedrock evaluation...")
            bedrock_eval = evaluator.custom_prompt_evaluation(
                generated_text, 
                custom_prompt="Critically analyze the following text for coherence and accuracy:",
                bedrock_model='claude-3-Opus'
            )
            print("\nBedrock Evaluation:")
            print(bedrock_eval['evaluation'])

            # print("\nAttempting Bedrock evaluation default ...")
            # bedrock_eval = evaluator.custom_prompt_evaluation(
            #     generated_text, 
            #     custom_prompt=None,
            #     bedrock_model='claude-3-Opus'
            # )
            # print("\nBedrock Evaluation:")
            # print(bedrock_eval['evaluation'])
        except Exception as e:
            print("\nBedrock evaluation skipped: Ensure AWS credentials are properly configured")
            print(f"Error: {str(e)}")

    
        # # Context evaluation example
        # question = "What are the main themes in Romeo and Juliet?"
        # context = """
        # Romeo and Juliet is a tragedy written by William Shakespeare. The play explores 
        # several major themes including love, fate, and conflict. The story revolves around 
        # two young lovers from feuding families, highlighting the destructive nature of 
        # family conflicts and the transformative power of love. The theme of fate is 
        # evident throughout the play, as various circumstances lead to the tragic ending.
        # """
        # answer = """
        # The main themes in Romeo and Juliet include the power of love, the destructive 
        # nature of family feuds, and the role of fate in human lives. The play shows how 
        # love can transcend social barriers but also how family conflicts can lead to 
        # tragic consequences.
        # """

        # # Evaluate context using different prompt types
        # context_eval = evaluator.evaluate_context(
        #     question=question,
        #     context=context,
        #     answer=answer,
        #     prompt_types=['relevance', 'factual_consistency', 'information_depth']
        # )


        # print("\nContext Evaluation Scores:")
        # if 'scores' in context_eval:
        #     for prompt_type, scores in context_eval['scores'].items():
        #         print(f"\n{prompt_type.upper()} Scores:")
        #         for metric, score in scores.items():
        #             print(f"  {metric}: {score}")
        
        # print("\nDetailed Evaluation Results:")
        # if 'detailed_results' in context_eval:
        #     for prompt_type, result in context_eval['detailed_results'].items():
        #         print(f"\n{prompt_type.upper()} Evaluation:")
        #         print(result['raw_evaluation'])

        # # Add a custom context evaluation prompt
        # custom_prompt = """
        # Analyze the emotional resonance of the context with respect to the question.
        
        # Question: {question}
        # Context: {context}
        
        # Evaluate:
        # 1. Emotional Depth: How well does the context convey the emotional aspects?
        # 2. Character Motivation: Are the character motivations clear?
        # 3. Thematic Impact: How effectively are the themes presented?
        
        # Rate each aspect from 1-10 and provide examples.
        # """
        
        # prompt_name = evaluator.add_custom_context_prompt(custom_prompt)
        # if prompt_name:
        #     custom_eval = evaluator.evaluate_context(
        #         question=question,
        #         context=context,
        #         prompt_types=[prompt_name]
        #     )
        #     print("\nCustom Context Evaluation Results:")
        #     print(custom_eval)

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
