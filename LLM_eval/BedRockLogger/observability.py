# Observability and Evaluation Custom Solution for Amazon Bedrock Applications
import pytz
import json
import time
import boto3
from uuid import uuid4
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from botocore.exceptions import ClientError

class BedrockLogs:
    VALID_FEATURE_NAMES = ["None", "Agent", "KB", "InvokeModel"]

    def __init__(self, delivery_stream_name: str = None, 
                 experiment_id: str = None, 
                 default_call_type: str = 'LLM', 
                 feature_name: str = None, 
                 feedback_variables: bool = False,
                 s3_bucket_name: str = None,
                 s3_region: str = "us-east-1"
                ):
        self.delivery_stream_name = delivery_stream_name
        self.experiment_id = experiment_id
        self.default_call_type = default_call_type
        self.feedback_variables = feedback_variables
        self.s3_bucket_name = s3_bucket_name
        self.s3_region = s3_region

        if feature_name is not None:
            if feature_name not in BedrockLogs.VALID_FEATURE_NAMES:
                raise ValueError(f"Invalid feature_name '{feature_name}'. Valid values are: {', '.join(BedrockLogs.VALID_FEATURE_NAMES)}")
        self.feature_name = feature_name
        self.step_counter = 0

        if self.experiment_id is None:
            self.experiment_id = 'default_experiment_1'

        if self.delivery_stream_name is None:
            raise ValueError("delivery_stream_name must be provided or set equals to 'local' example: delivery_stream_name='local'.")

        if self.s3_bucket_name is None:
            raise ValueError("s3_bucket_name must be provided if streaming is set equals to 's3' example: delivery_stream_name='s3'.")

        if self.delivery_stream_name == 'local':
            self.firehose_client = None
        elif self.delivery_stream_name == 's3':
            self.s3_client = boto3.client('s3', region_name=self.s3_region)
            self.ensure_bucket_exists(self.s3_bucket_name, self.s3_region)
        else:
            self.firehose_client = boto3.client('firehose')

    @staticmethod
    def find_keys(dictionary, key, path=[]):
        """
        Recursive function to find all keys in a nested dictionary and their paths.

        Args:
            dictionary (dict): The dictionary to search.
            key (str): The key to search for.
            path (list, optional): The path of keys to the current dictionary. Defaults to None.

        Returns:
            list: A list of tuples containing the key's path and value.
        """
        results = []

        if isinstance(dictionary, dict):
            for k, v in dictionary.items():
                new_path = path + [k]
                if k == key:
                    results.append((new_path, v))
                else:
                    results.extend(BedrockLogs.find_keys(v, key, new_path))
        elif isinstance(dictionary, list):
            for i, item in enumerate(dictionary):
                new_path = path + [i]
                results.extend(BedrockLogs.find_keys(item, key, new_path))

        return results

    def ensure_bucket_exists(self, bucket_name, aws_region="us-east-1"):
        try:
            # Check if the bucket exists
            self.s3_client.head_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' already exists.")
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                # Bucket does not exist, create it
                print(f"Bucket '{bucket_name}' does not exist. Creating...")
                if aws_region == "us-east-1":
                    self.s3_client.create_bucket(Bucket=bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": aws_region}
                    )
                print(f"Bucket '{bucket_name}' created successfully.")
            else:
                raise Exception(f"Failed to check or create bucket: {e}")

    def save_log_to_s3(self, metadata, object_key):
        try:
            json_string = json.dumps(metadata)
            self.s3_client.put_object(
                Bucket=self.s3_bucket_name,
                Key=object_key,
                Body=json_string,
                ContentType="application/json"
            )
            print(f"Log saved to S3: s3://{self.s3_bucket_name}/{object_key}")
        except Exception as e:
            raise Exception(f"Failed to save log to S3: {str(e)}")


    def extract_session_id(self, log_data: Dict[str, Any]) -> str:
        """
        Extracts the session ID from the log data. If the session ID is not available,
        it generates a new UUID for the run ID.

        Args:
            log_data (Dict[str, Any]): The log data dictionary.

        Returns:
            str: The session ID or a newly generated UUID if the session ID is not available.
        """
        if self.feature_name == "Agent":
            session_id_paths = self.find_keys(log_data, 'x-amz-bedrock-agent-session-id')
        else:
            session_id_paths = self.find_keys(log_data, 'sessionId')

        if session_id_paths:
            path, session_id = session_id_paths[0]
            return session_id
        else:
            return str(uuid4())

    def handle_agent_feature(self, output_data, request_start_time):
        """
        Handles the logic for the 'Agent' feature, including step counting and latency calculation.

        Args:
            output_data (Any): The output data from the function call.
            request_start_time (float): The start time of the request.

        Returns:
            Any: The updated output data with step numbers and latency information.
        """
        self.session_id = None
        prev_trace_time = None
        for data in output_data:
            if isinstance(data, dict) and 'trace' in data:
                trace = data['trace']
                if 'start_trace_time' in trace:
                    # Check if 'start_trace_time' is defined correctly
                    if not isinstance(trace['start_trace_time'], float):
                        raise ValueError("The key 'start_trace_time' should be present and should be a time.time() object.")

                    # Calculate the latency between traces
                    if prev_trace_time is None:
                        trace['latency'] = trace['start_trace_time'] - request_start_time
                    else:
                        trace['latency'] = trace['start_trace_time'] - prev_trace_time

                    prev_trace_time = trace['start_trace_time']
                    trace['step_number'] = self.step_counter
                    self.step_counter += 1
                    data['trace'] = trace  # Update the 'trace' dictionary in the original data

            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'start_trace_time' in item:
                        # Check if 'start_trace_time' is defined correctly
                        if not isinstance(item['start_trace_time'], float):
                            raise ValueError("The key 'start_trace_time' should be present and should be a time.time() object.")

                        # Calculate the latency between traces
                        if prev_trace_time is None:
                            item['latency'] = item['start_trace_time'] - request_start_time
                        else:
                            item['latency'] = item['start_trace_time'] - prev_trace_time

                        prev_trace_time = item['start_trace_time']
                        item['step_number'] = self.step_counter
                        self.step_counter += 1

                    elif isinstance(item, dict) and 'trace' in item:
                        trace = item['trace']
                        if 'start_trace_time' in trace:
                            # Check if 'start_trace_time' is defined correctly
                            if not isinstance(trace['start_trace_time'], float):
                                raise ValueError("The key 'start_trace_time' should be present and should be a time.time() object.")

                            # Calculate the latency between traces
                            if prev_trace_time is None:
                                trace['latency'] = trace['start_trace_time'] - request_start_time
                            else:
                                trace['latency'] = trace['start_trace_time'] - prev_trace_time

                            prev_trace_time = trace['start_trace_time']
                            trace['step_number'] = self.step_counter
                            self.step_counter += 1
                            item['trace'] = trace  # Update the 'trace' dictionary in the original item

        return output_data

    def watch(self, capture_input: bool = True, capture_output: bool = True, call_type: Optional[str] = None):
        def wrapper(func):
            def inner(*args, **kwargs):
                # For Latency Calculation:
                self.request_start_time = time.time()

                # Get the function name
                function_name = func.__name__

                # Capture input if requested
                input_data = args if capture_input else None
                input_log = None
                if input_data:
                    input_log = input_data[0]
                    
                # Generate observation_id
                observation_id = str(uuid4())
                obs_timestamp = datetime.now(timezone.utc).isoformat()

                # Get the start time
                start_time = time.time()

                # Calls the function to be executed
                result = func(*args, **kwargs)
                
                # Capture output if requested
                output_data = result if capture_output else None

                # Get the end time
                end_time = time.time()

                # Calculate the duration
                duration = end_time - start_time
                
                # Begin Logging Time:
                logging_start_time = time.time()

                # Handle the 'Agent' feature case
                if self.feature_name == "Agent":
                    if output_data is not None:
                        output_data = self.handle_agent_feature(output_data, self.request_start_time)
                        run_id = self.extract_session_id(output_data[0])
                    else:
                        run_id = self.extract_session_id(input_log)
                else:
                    # Extract the session ID from the log or generate a new one
                    run_id = self.extract_session_id(input_log)

                # Prepare the metadata
                metadata = {
                    'experiment_id': self.experiment_id,
                    'run_id': run_id,
                    'observation_id': observation_id,
                    'obs_timestamp': obs_timestamp,
                    'start_time': datetime.fromtimestamp(start_time, tz=pytz.utc).isoformat(),
                    'end_time': datetime.fromtimestamp(end_time, tz=pytz.utc).isoformat(),
                    'duration': duration,
                    'input_log': input_log,
                    'output_log': output_data,
                    'call_type': call_type or self.default_call_type,
                    'feature_name': self.feature_name,
                    'feedback_enabled': self.feedback_variables
                }

                # Update the metadata with additional_metadata if provided
                additional_metadata = kwargs.get('additional_metadata', {})
                if additional_metadata:
                    metadata.update(additional_metadata)

                input_data = kwargs.get('user_prompt', {})
                if input_data:
                    metadata.update(input_data)
                    
                # Get the end time
                logging_end_time = time.time()

                # Calculate the duration
                logging_duration = logging_end_time - logging_start_time
                metadata['logging_duration'] = logging_duration

                # Send the metadata to Amazon  S3 , Kinesis Data Firehose or return it locally for testing:
                if self.delivery_stream_name == 'local':
                    if self.feedback_variables:
                        print("Logs in local mode-with feedback:")
                        return result, metadata
                    else:
                        print("Logs in local mode-without feedback:")
                        return result, metadata

                elif self.delivery_stream_name == 's3':
                    # Save log to S3
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    object_key = f"logs/{self.experiment_id}/{timestamp}_log.json"
                    self.save_log_to_s3(metadata, object_key)
                    if self.feedback_variables:
                        print("Logs in S3-with feedback:")
                        return result, metadata
                    else:
                        print("Logs in S3-without feedback:")
                        return result
                # log to firehose
                else:
                    firehose_response = self.firehose_client.put_record(
                        DeliveryStreamName=self.delivery_stream_name,
                        Record={
                            'Data': json.dumps(metadata)
                        }
                    )
                    if self.feedback_variables:
                        print("Logs in S3-with feedback:")
                        return result, run_id, observation_id
                    else:
                        print("Logs in S3-without feedback:")
                        return result

            return inner
        return wrapper
