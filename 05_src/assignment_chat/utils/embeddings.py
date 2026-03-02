import os
from openai import OpenAI
from chromadb import EmbeddingFunction, Documents, Embeddings

_GATEWAY_URL = 'https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1'


class GatewayEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding function routed through the AWS API Gateway."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self._client = OpenAI(
            base_url=_GATEWAY_URL,
            api_key='any value',
            default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
        )
        self._model = model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = self._client.embeddings.create(model=self._model, input=input)
        return [e.embedding for e in response.data]
