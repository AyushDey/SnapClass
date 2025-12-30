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
    from main import app
    # We need to import the global classifier variable, but it's None until startup.
    # However, TestClient(app) triggers startup. 
    # But because we need to patch the classifier's dir *after* startup but *before* tests (or simpler: patch it on the instance),
    # let's modify the flow.

    with TestClient(app) as client:
        # Import inside to get the updated reference if needed, 
        # but 'from main import classifier' imports the name at that moment (which is None).
        # We need to access it from the module namespace dynamically or after startup.
        import main
        
        if main.classifier is None:
            raise RuntimeError("Classifier not initialized after TestClient startup")

        # Save original dir
        original_dir = main.classifier.references_dir
        
        # Point to temp dir
        main.classifier.references_dir = temp_references_dir
        # Reset internal state
        main.classifier.reference_embeddings = {}
        main.classifier.search_matrix = None
        main.classifier.search_labels = []
        
        yield client
        
        # Restore
        if main.classifier:
            main.classifier.references_dir = original_dir
