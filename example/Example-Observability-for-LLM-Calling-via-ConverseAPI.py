import ai_monitor
from ai_monitor import BedrockLogs
import boto3
import json

# Initialize BedrockLogs in Local mode with feedback variables
bedrock_logs = BedrockLogs(delivery_stream_name='local', feedback_variables=True,s3_bucket_name='logging-response',s3_region='us-east-1')

@bedrock_logs.watch(call_type='Converse-API')
def get_summary(context):

  """ Get the summary of the data """
  model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
  region_name = 'us-east-1'
  guardrail_id = "gf1n57xgv52p"
  guardrail_version = "7"
  session = boto3.Session()
  bedrock_agent_runtime_client = session.client("bedrock-runtime", region_name=region_name)
  model_arn = f'arn:aws:bedrock:{region_name}::foundation-model/{model_id}'

  # context = query
  temperature=0.5 
  top_p=1.0
  top_k=32
  candidate_count=1
  max_output_tokens=2048

  user_message = context

  conversation = [
      {
          "role": "user",
          "content": [{"text": user_message} ],
      }
    ]

  guardrail_config = {
      "guardrailIdentifier": guardrail_id,
      "guardrailVersion": guardrail_version,
      "trace": "enabled"
  }
  try:
    # Send the message to the model, using a basic inference configuration.
    response = bedrock_agent_runtime_client.converse(
      modelId=model_id,
      messages=conversation,
      inferenceConfig={"maxTokens": max_output_tokens, "temperature": temperature, "topP": top_p},
      additionalModelRequestFields={},
      guardrailConfig=guardrail_config
    )

    # print(response)
    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]

    # print(response_text)

    return response

  except Exception as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    return (f"ERROR: Can't invoke {model_id}. Reason: {e}")

response , metadata = get_summary("what is python")

response_text = response["output"]["message"]["content"][0]["text"]

# print(response_text)

print(metadata)

config_filename = "metadata.json"
with open(config_filename, "w") as config_file:
    json.dump(metadata, config_file, indent=4)
