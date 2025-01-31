import pytest
from unittest.mock import patch, MagicMock
from aiklyra.client import (
    AiklyraClient,
    InvalidAPIKeyError,
    InsufficientCreditsError,
    AnalysisError,
    AiklyraAPIError,
    ConversationFlowAnalysisRequest,
    ConversationFlowAnalysisResponse,
    ValidationError,  
)


@pytest.fixture
def setup_client():
    """
    Fixture to set up test client and data.
    """
    api_key = "test_api_key"
    base_url = "http://localhost:8002"
    client = AiklyraClient(api_key=api_key, base_url=base_url)
    conversation_data = {
        "session_1": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    return client, conversation_data


@patch("aiklyra.client.requests.post")
def test_analyse_success(mock_post, setup_client):
    """
    Test successful analysis.
    """
    client, conversation_data = setup_client

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "transition_matrix": [[0.1, 0.9], [0.8, 0.2]],
        "intent_by_cluster": {0: "greeting", 1: "response"},
    }
    mock_post.return_value = mock_response

    result = client.analyse(conversation_data)

    assert isinstance(result, ConversationFlowAnalysisResponse)
    assert result.transition_matrix == [[0.1, 0.9], [0.8, 0.2]]
    assert result.intent_by_cluster[0] == "greeting"

    # Ensure Authorization header is included in the request
    mock_post.assert_called_once_with(
        f"{client.base_url}/{AiklyraClient.BASE_ANALYSE_ENDPOINT}",
        headers={
            "Authorization": f"Bearer {client.api_key}",
            "Content-Type": "application/json",
        },
        json=ConversationFlowAnalysisRequest(conversation_data=conversation_data).model_dump(),
    )


@patch("aiklyra.client.requests.post")
def test_analyse_invalid_api_key(mock_post, setup_client):
    """
    Test invalid API key error.
    """
    client, conversation_data = setup_client

    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.json.return_value = {"detail": "Invalid API Key"}
    mock_post.return_value = mock_response

    with pytest.raises(InvalidAPIKeyError):
        client.analyse(conversation_data)


@patch("aiklyra.client.requests.post")
def test_analyse_insufficient_credits(mock_post, setup_client):
    """
    Test insufficient credits error.
    """
    client, conversation_data = setup_client

    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.json.return_value = {"detail": "Insufficient credits"}
    mock_post.return_value = mock_response

    with pytest.raises(InsufficientCreditsError):
        client.analyse(conversation_data)


@patch("aiklyra.client.requests.post")
def test_analyse_analysis_error(mock_post, setup_client):
    """
    Test analysis error with malformed response.
    """
    client, conversation_data = setup_client

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"invalid_key": "unexpected_data"}
    mock_post.return_value = mock_response

    with pytest.raises(AnalysisError):
        client.analyse(conversation_data)


@patch("aiklyra.client.requests.post")
def test_analyse_api_error(mock_post, setup_client):
    """
    Test generic API error.
    """
    client, conversation_data = setup_client

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response

    with pytest.raises(AiklyraAPIError) as exc_info:
        client.analyse(conversation_data)

    assert "Error 500" in str(exc_info.value)


@patch("aiklyra.client.requests.post")
def test_missing_authorization_header(mock_post, setup_client):
    """
    Test missing Authorization header.
    """
    client, conversation_data = setup_client

    mock_response = MagicMock()
    mock_response.status_code = 422
    mock_response.json.return_value = {
        "detail": [
            {"type": "missing", "loc": ["header", "authorization"], "msg": "Field required"}
        ]
    }
    mock_post.return_value = mock_response

    with pytest.raises(AiklyraAPIError) as exc_info:
        client.analyse(conversation_data)

    assert "Error 422" in str(exc_info.value)
@patch("aiklyra.client.requests.post")
def test_analyse_invalid_conversation_data_type(mock_post, setup_client):
    """Test ValidationError when conversation_data is not a dictionary."""
    client, _ = setup_client
    invalid_conversation_data = ["not a dictionary"]
    
    with pytest.raises(ValidationError) as exc_info:
        client.analyse(conversation_data=invalid_conversation_data)
    
    assert "conversation_data must be a dictionary." in str(exc_info.value)
    mock_post.assert_not_called()


@patch("aiklyra.client.requests.post")
def test_analyse_invalid_min_clusters(mock_post, setup_client):
    """Test ValidationError when min_clusters is non-positive."""
    client, valid_data = setup_client
    
    with pytest.raises(ValidationError) as exc_info:
        client.analyse(conversation_data=valid_data, min_clusters=0)
    
    assert "min_clusters and max_clusters must be positive integers." in str(exc_info.value)
    mock_post.assert_not_called()


@patch("aiklyra.client.requests.post")
def test_analyse_invalid_max_clusters(mock_post, setup_client):
    """Test ValidationError when max_clusters is non-positive."""
    client, valid_data = setup_client
    
    with pytest.raises(ValidationError) as exc_info:
        client.analyse(conversation_data=valid_data, max_clusters=-5)
    
    assert "min_clusters and max_clusters must be positive integers." in str(exc_info.value)
    mock_post.assert_not_called()


@patch("aiklyra.client.requests.post")
def test_analyse_min_clusters_greater_than_max(mock_post, setup_client):
    """Test ValidationError when min_clusters exceeds max_clusters."""
    client, valid_data = setup_client
    
    with pytest.raises(ValidationError) as exc_info:
        client.analyse(
            conversation_data=valid_data,
            min_clusters=10,
            max_clusters=5
        )
    
    assert "Max clusters needs to be greater than Min Clusters" in str(exc_info.value)
    mock_post.assert_not_called()