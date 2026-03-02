import pytest
import os
import shutil
from fastapi.testclient import TestClient
from db import sessionLocal, engine
from models import Base

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

@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    
    from sqlalchemy import delete
    from models import BookletItem
    
    # Clean before test
    with sessionLocal() as session:
        session.execute(delete(BookletItem))
        session.commit()
        
    yield
    
    # Clean after test
    with sessionLocal() as session:
        session.execute(delete(BookletItem))
        session.commit()

@pytest.fixture
def api_client(temp_references_dir):
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
        old_classifier = main.classifier
        
        # Create a fresh classifier instance for this test session context
        from classifier import ImageClassifier
        main.classifier = ImageClassifier(references_dir=temp_references_dir, session_factory=sessionLocal)
        
        yield client
        
        # Restore original classifier
        main.classifier = old_classifier
