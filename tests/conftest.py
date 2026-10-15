from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_heavy_models():
    """Mocks SentenceTransformer, HuggingFace pipeline, and QdrantClient

    during unit tests to prevent downloading models in CI.
    """
    with (
        patch("legal_rag.core.rag.SentenceTransformer") as mock_transformer,
        patch("legal_rag.core.rag.pipeline") as mock_pipeline,
        patch("legal_rag.core.rag.QdrantClient") as mock_qdrant,
    ):
        # Configure mock encoder return shapes
        mock_encoder_instance = MagicMock()
        mock_encoder_instance.encode.return_value.tolist.return_value = [0.1] * 1024
        mock_transformer.return_value = mock_encoder_instance

        # Configure mock LLM pipeline return format
        mock_llm_instance = MagicMock()
        mock_llm_instance.tokenizer.apply_chat_template.return_value = "Prompt"
        mock_llm_instance.return_value = [{"generated_text": "Sample answer"}]
        mock_pipeline.return_value = mock_llm_instance

        # Configure mock Qdrant collection info
        mock_qdrant_instance = MagicMock()
        mock_qdrant_instance.get_collection.return_value.points_count = 100
        mock_qdrant.return_value = mock_qdrant_instance

        yield