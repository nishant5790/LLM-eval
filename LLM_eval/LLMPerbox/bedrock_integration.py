import boto3
import json
import ast
class BedrockEvaluator:
    def __init__(self, region_name='us-east-1'):
        """
        Initialize Bedrock client
        
        :param region_name: AWS region for Bedrock service
        """
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        
        # Predefined Bedrock model configurations
        self.models = {
            'claude-3-Opus': 'us.anthropic.claude-3-opus-20240229-v1:0',
            'claude-3-5-haiku': 'us.anthropic.claude-3-5-haiku-20241022-v1:0',
            'claude-3-5-sonnet': 'us.anthropic.claude-3-5-sonnet-20240620-v1:0',
            'llama3-2-11b': 'us.meta.llama3-2-11b-instruct-v1:0'

        }

        self.embedding_models = {
            'amazon.titan-embed-text-v2': 'amazon.titan-embed-text-v2:0',
            'amazon.titan-embed-text-v1': 'amazon.titan-embed-text-v1'
        }

    def select_model(self, model_name='claude-3-haiku'):
        """
        Select a Bedrock model for evaluation
        
        :param model_name: Name of the Bedrock model
        :return: Model ID
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.models.keys())}")
        return self.models[model_name]

    def model_invoke(self, modelId,prompt,modelConfig=None):
        """
        Invoke a Bedrock model for evaluation
        
        :param modelId: Model ID
        :param prompt: Evaluation prompt
        :param modelconfig: Model configuration
        :return: Evaluation result
        """
        if modelId is None:
            raise ValueError("Model ID cannot be empty")
        if prompt is None:
            raise ValueError("Prompt cannot be empty")

        if modelConfig is None:
            temperature=0
            top_p=1.0
            max_output_tokens=2048

            if modelConfig is not None:
                temperature=modelConfig['temperature']
                top_p=modelConfig['top_p']
                max_output_tokens=modelConfig['max_output_tokens']

            payload = {
                "textGenerationConfig": {
                    "temperature": temperature,
                    "topP": top_p,
                    "maxTokens": max_output_tokens
                },

                "conversation" :[
                    {
                        "role": "user",
                        "content": [{"text": prompt} ]
                    }
                ]

            }
        try:
            response = self.bedrock_runtime.converse(
                modelId=modelId,
                messages=payload['conversation'],
                inferenceConfig = payload['textGenerationConfig'],
            )
            return response["output"]["message"]["content"][0]["text"] , response
        except Exception as e:
            print(e)
            return {'error': "error at model_invoke: " + str(e)}

    def invoke_embedding(self, text, model_name='amazon.titan-embed-text-v2'):
        """
        Invoke a Bedrock model for text embedding
        
        :param text: Text to embed
        :param model_name: Embedding model name
        :return: Embedding result
        """
        model_id = self.embedding_models[model_name]
        if model_id is None:
            raise ValueError("Model ID cannot be empty")

        body=json.dumps({
            "inputText": text
        })
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body,
                trace='ENABLED',
                accept = 'application/json'
            )
            response_body = json.loads(response.get('body').read())
            return response_body['embedding']
        except Exception as e:
            return {'error': str(e)}

    def evaluate_with_prompt(self, generated_text, evaluation_prompt=None, model_name='claude-3-Opus'):
        """
        Evaluate generated text using a Bedrock model and custom prompt
        
        :param generated_text: Text to evaluate
        :param evaluation_prompt: Custom prompt for evaluation (optional)
        :param model_name: Bedrock model to use
        :return: Evaluation result
        """
        # Default evaluation prompt if not provided
        evaluation_prompt_input = evaluation_prompt
        if not evaluation_prompt_input:
            evaluation_prompt = f"""Evaluate the following text critically:
            Text: {generated_text}
            
            Please provide a detailed assessment considering:
            1. Coherence
            2. Relevance
            3. Factual accuracy
            4. Potential biases
            
            Give a comprehensive score and explanation and IMPORTANT: Please make sure to only return in JSON format. JSON format which will have following :
            ''' 
                "coherence": coherence_score betweenn 1 to 10,
                "coherence_explanation": The text is coherent and grammatically correct  docstring format,
                "relevance": relevance_score betweenn 1 to 10,
                "relevance_explanation": The text is relevant to the question  docstring format,
                "factual_accuracy": relevance_score betweenn 1 to 10,
                "factual_accuracy_explanation": The text provides accurate information  docstring format,
                "potential_biases": potential_biases_score betweenn 1 to 10,
                "potential_biases_explanation": The text does not contain any biases in docstring format
            '''
            """
        else:
            evaluation_prompt=  evaluation_prompt  + f""" Given Generated Text : {generated_text} , give a comprehensive score and explanation
            IMPORTANT: Please make sure to only return in JSON format 
            """
            
        # Prepare the request payload
        model_id = self.select_model(model_name)
        
        try:
            # Invoke Bedrock model
            response , metadata = self.model_invoke(
                modelId=model_id,
                prompt=evaluation_prompt,
                modelConfig=None
            )
        except Exception as e:
            print(e)
            return {
                'error': str(e),
                'model': model_name
            }
        
        # Parse and return the evaluation result
        if not evaluation_prompt_input:
            try:
                response = ast.literal_eval(response)
            except Exception as e:
                response = {"WARNING": f"Failed to parse the response: {e}",
                            "model_response": response}

        return {
            'model': model_name,
            'evaluation': response,
            'metadata': metadata
        }
        
