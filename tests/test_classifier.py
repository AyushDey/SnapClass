import torch
import pytest
from PIL import Image
from classifier import ImageClassifier
from db import sessionLocal, engine
from models import Base

class TestImageClassifier:
    
    @pytest.fixture
    def db_session_factory(self):
        # We use the existing postgres engine to support pgvector Vector types
        Base.metadata.create_all(bind=engine)
        return sessionLocal

    @pytest.fixture
    def classifier(self, temp_references_dir, db_session_factory):
        # Initialize with temp dir and real db session for pgvector support
        return ImageClassifier(references_dir=temp_references_dir, session_factory=db_session_factory)

    def test_initialization(self, classifier):
        assert classifier.model is not None
        assert classifier.device.type == "cpu"
        assert classifier.search_matrix is None

    def test_classify_no_references(self, classifier, mock_image):
        # Should handle graceful failure
        result = classifier.classify(mock_image)
        assert result["class"] == "Unknown"
        assert result["confidence"] == pytest.approx(0.0)
        assert result["message"] == "No references available"

    def test_classify_flow(self, classifier, mock_image):
        # Inject fake reference
        with classifier._lock:
            # Create a fake reference embedding identical to mock_image's embedding
            target_emb = classifier.get_embedding(mock_image)
            
            # Add noise to make a second class
            noise_emb = torch.randn(1, 512)
            noise_emb = torch.nn.functional.normalize(noise_emb, p=2, dim=1)
            
            classifier.search_matrix = torch.cat([target_emb, noise_emb])
            classifier.search_labels = ["target_class", "noise_class"]
            
            # Use integer IDs for categories to match the new database schema logic
            classifier.search_categories = [1, 1] 
            
        result = classifier.classify(mock_image, threshold=0.5)
        
        assert result["class"] == "target_class"
        assert len(result["matches"]) == 1
        assert result["matches"][0]["class"] == "noise_class"