import pytest
import torch
from pathlib import Path
from classifier import ImageClassifier
from PIL import Image

@pytest.fixture
def db_session_factory():
    from db import sessionLocal, engine
    from models import Base
    Base.metadata.create_all(bind=engine)
    return sessionLocal

@pytest.fixture
def temp_references_dir(tmp_path):
    path = tmp_path / "references"
    path.mkdir()
    return str(path)

def test_persistence(db_session_factory, temp_references_dir):
    class TestClassifier(ImageClassifier):
        def __init__(self):
            super().__init__(references_dir=temp_references_dir, session_factory=db_session_factory)

    # Use Pathlib to create files
    label_dir = Path(temp_references_dir) / "cat"
    label_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (64, 64), color='red')
    img_path = label_dir / "cat1.png"
    img.save(img_path)

    clf1 = TestClassifier()
    assert "cat" in clf1.search_labels
    assert len(clf1.search_labels) > 0
    
    from models import BookletItem
    from sqlalchemy import select
    with db_session_factory() as session:
        stmt = select(BookletItem)
        items = session.execute(stmt).scalars().all()
        assert len(items) > 0
    
    clf2 = TestClassifier()
    assert "cat" in clf2.search_labels
    assert len(clf2.search_labels) == len(clf1.search_labels)
    assert torch.allclose(clf1.search_matrix, clf2.search_matrix)

def test_hashing_update(db_session_factory, temp_references_dir):
    class TestClassifier(ImageClassifier):
        def __init__(self):
            super().__init__(references_dir=temp_references_dir, session_factory=db_session_factory)

    label_dir = Path(temp_references_dir) / "dog"
    label_dir.mkdir(parents=True, exist_ok=True)
    img_path = label_dir / "dog1.png"
    Image.new('RGB', (10, 10), color='white').save(img_path)

    clf1 = TestClassifier()
    count1 = len(clf1.search_labels)
    
    Image.new('RGB', (10, 10), color='black').save(img_path)
    clf2 = TestClassifier()
    
    assert len(clf2.search_labels) == count1