import pytest
import os
import shutil
from fastapi.testclient import TestClient

@pytest.fixture
def temp_references_dir(tmp_path):
    """Creates a temporary directory for references using pytest's tmp_path."""
    # tmp_path is unique for each test invocation
    path = tmp_path / "references"
    path.mkdir()
    return str(path)

@pytest.fixture
def mock_image():
    """Returns a simple red generic image."""
    from PIL import Image
    return Image.new('RGB', (224, 224), color='red')

@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Creates a temporary directory for ChromaDB."""
    path = tmp_path / "chroma_db"
    # Chroma creates the dir if it doesn't exist, passing the path str is enough
    return str(path)

@pytest.fixture
def api_client(temp_references_dir, temp_chroma_dir):
    """
    Returns a TestClient where the global classifier is pointed 
    to temporary directories to avoid messing with real data.
    """
    from main import app
    # We need to import the global classifier variable, but it's None until startup.
    # TestClient(app) triggers startup. 

    with TestClient(app) as client:
        import main
        
        if main.classifier is None:
            raise RuntimeError("Classifier not initialized after TestClient startup")

        # Save original state
        # Only saving the client path, assuming we can swap it out. 
        # Chroma client is persistent, so we might need to re-init or just swap the path if logic allows.
        # But `classifier.py` initializes client in `__init__`.
        # To strictly isolate, we should likely re-initialize the classifier or 
        # minimalistically, just ensure we don't write to the old DB.
        
        # Ideally we'd re-init the whole classifier, but `app.state` or global var?
        # It's a global var. Let's try to re-instantiate it for the test.
        
        old_classifier = main.classifier
        
        # Create a fresh classifier instance for this test session context
        # This ensures it connects to the TEMP chroma db
        from classifier import ImageClassifier
        main.classifier = ImageClassifier(references_dir=temp_references_dir, chroma_db_path=temp_chroma_dir)
        
        yield client
        
        # Restore original classifier
        main.classifier = old_classifier
