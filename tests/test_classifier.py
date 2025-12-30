import torch
import pytest
from PIL import Image
from classifier import ImageClassifier

class TestImageClassifier:
    
    @pytest.fixture
    def classifier(self, temp_references_dir):
        # Initialize with temp dir
        return ImageClassifier(references_dir=temp_references_dir)

    def test_initialization(self, classifier):
        assert classifier.model is not None
        assert classifier.device.type == "cpu"
        assert classifier.search_matrix is None

    def test_get_scaled_embedding_logic(self, classifier, mock_image):
        # Scale 1.0 (Exact)
        emb_1 = classifier._get_scaled_embedding(mock_image, 1.0)
        assert emb_1.shape == (1, 512)
        
        # Scale 0.5 (Resize)
        emb_05 = classifier._get_scaled_embedding(mock_image, 0.5)
        assert emb_05.shape == (1, 512)
        # Verify it handled resizing without crashing

    def test_collect_matches_logic(self, classifier):
        # Mock search matrix: 3 items
        # Item 0: [1, 0, 0]
        # Item 1: [0, 1, 0]
        # Item 2: [0, 0, 1]
        
        # Note: Embedding size 512, simplifying for test logic if possible, 
        # but function expects shapes to match. 
        # We'll use 512 dims but simple vectors.
        dim = 512
        matrix = torch.zeros(3, dim)
        matrix[0, 0] = 1.0 # Vector A
        matrix[1, 1] = 1.0 # Vector B
        matrix[2, 2] = 1.0 # Vector C
        
        # Search for Vector A (should match Item 0 best)
        target = torch.zeros(1, dim)
        target[0, 0] = 1.0
        
        matches = classifier._collect_matches_from_embedding(target, matrix)
        
        # Matches is list of (score, index)
        # Expected: Index 0 has score 1.0
        
        top_score, top_idx = matches[0]
        assert top_idx == 0
        assert pytest.approx(top_score, 0.001) == 1.0

    def test_classify_no_references(self, classifier, mock_image):
        # Should handle graceful failure
        result = classifier.classify(mock_image)
        assert result["class"] == "Unknown"
        assert result["confidence"] == 0.0
        assert result["message"] == "No references available"

    def test_classify_flow(self, classifier, mock_image):
        # Inject fake reference
        with classifier._lock:
            # Create a fake reference embedding identical to mock_image's embedding
            # This ensures a perfect match logic test without file I/O
            target_emb = classifier.get_embedding(mock_image)
            
            # Add noise to make a second class
            noise_emb = torch.randn(1, 512)
            noise_emb = torch.nn.functional.normalize(noise_emb, p=2, dim=1)
            
            classifier.search_matrix = torch.cat([target_emb, noise_emb])
            classifier.search_labels = ["target_class", "noise_class"]
            
        result = classifier.classify(mock_image, threshold=0.5)
        
        assert result["class"] == "target_class"
        # Since we excluded the winner from matches in previous step, check that:
        assert len(result["matches"]) == 1
        assert result["matches"][0]["class"] == "noise_class"
