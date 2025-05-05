import json
import boto3
import asyncio
from botocore.exceptions import ClientError
from config import AWS_REGION, EMBEDDING_MODEL_ID, LLM_MODEL_ID
from utils.logger import logger

# Initialize a shared boto3 client for Bedrock service
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def generate_embedding(text: str) -> list:
    """
    Generate embeddings for text using AWS Bedrock's embedding model
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
    try:
        max_tokens = 8000  # Embedding model input token limit
        tokens = text.split()  # Simple tokenization by spaces
        text = " ".join(tokens[:max_tokens])  # Keep only allowed tokens
        payload = {"inputText": text}
        response = bedrock_client.invoke_model(
            modelId=EMBEDDING_MODEL_ID, body=json.dumps(payload)
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

async def send_to_bedrock(prompt):
    """
    Send a prompt to the Bedrock Claude model asynchronously.
    """
    payload = [{"role": "user", "content": [{"text": prompt}]}]
    model_id = LLM_MODEL_ID
    try:
        # Use asyncio.to_thread to call the blocking boto3 client method
        response = await asyncio.to_thread(
            bedrock_client.converse,
            modelId=model_id,
            messages=payload,
        )
        model_response = response["output"]["message"]
        # Concatenate text parts from the model response
        response_text = " ".join(i["text"] for i in model_response["content"])
        return response_text
    except ClientError as err:
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        raise