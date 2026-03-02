import io
import zipfile
from pathlib import Path


def test_bulk_upload_creates_labels(api_client, temp_references_dir):
    """Test that bulk upload creates labels and images correctly."""
    # Create a ZIP file in memory with two label folders
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Create a simple PNG image bytes (1x1 red pixel)
        from PIL import Image
        
        for label in ["cat", "dog"]:
            for i in range(2):
                img = Image.new('RGB', (100, 100), color='red')
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                zf.writestr(f"{label}/image_{i}.png", img_buffer.read())
    
    zip_buffer.seek(0)
    
    # Upload the ZIP
    response = api_client.post(
        "/bulk_upload",
        files={"file": ("test_refs.zip", zip_buffer, "application/zip")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Bulk upload successful"
    assert set(data["labels"].keys()) == {"cat", "dog"}
    assert data["labels"]["cat"] == 2
    assert data["labels"]["dog"] == 2
    assert data["total_images"] == 4
    
    # Verify files on disk using Pathlib
    cat_dir = Path(temp_references_dir) / "cat"
    dog_dir = Path(temp_references_dir) / "dog"
    assert cat_dir.exists()
    assert dog_dir.exists()
    assert len(list(cat_dir.iterdir())) == 2
    assert len(list(dog_dir.iterdir())) == 2


def test_bulk_upload_invalid_file_type(api_client):
    """Test that non-archive files are rejected."""
    response = api_client.post(
        "/bulk_upload",
        files={"file": ("test.txt", b"some text", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


def test_bulk_upload_invalid_zip(api_client):
    """Test that corrupted archives return an error."""
    response = api_client.post(
        "/bulk_upload",
        files={"file": ("fake.zip", b"not a real zip", "application/zip")}
    )
    # Corrupted archives fail during extraction and return 500
    assert response.status_code == 500
    assert "Bulk upload failed" in response.json()["error"]


def test_bulk_upload_empty_zip(api_client, temp_references_dir):
    """Test that an empty ZIP returns zero images."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED):
        pass  # Empty ZIP
    
    zip_buffer.seek(0)
    
    response = api_client.post(
        "/bulk_upload",
        files={"file": ("empty.zip", zip_buffer, "application/zip")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_images"] == 0
    assert data["labels"] == {}