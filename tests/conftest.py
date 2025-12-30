import pytest
import os
import shutil
import tempfile
from PIL import Image
from fastapi.testclient import TestClient

@pytest.fixture
def temp_references_dir():
    """Creates a temporary directory for references and cleans it up after."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

@pytest.fixture
def mock_image():
    """Returns a simple red generic image."""
    return Image.new('RGB', (224, 224), color='red')

@pytest.fixture
def api_client(temp_references_dir):
    """
    Returns a TestClient where the global classifier is pointed 
    to a temporary directory to avoid messing with real data.
    """
    from main import app, classifier
    
    # Save original dir
    original_dir = classifier.references_dir
    
    # Point to temp dir
    classifier.references_dir = temp_references_dir
    # Reset internal state
    classifier.reference_embeddings = {}
    classifier.search_matrix = None
    classifier.search_labels = []
    
    client = TestClient(app)
    yield client
    
    # Restore
    classifier.references_dir = original_dir
