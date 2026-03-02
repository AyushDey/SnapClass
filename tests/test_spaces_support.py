import io
import zipfile
from pathlib import Path
from PIL import Image

def test_add_reference_with_spaces_in_label(api_client, temp_references_dir):
    """Test adding a reference with spaces in the label via API."""
    img = Image.new('RGB', (100, 100), color='blue')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    label_with_spaces = "blue sky"
    
    response = api_client.post(
        "/add_reference",
        data={"label": label_with_spaces},
        files={"file": ("sky.png", img_buffer, "image/png")}
    )
    
    assert response.status_code == 200
    sanitized_label = "blue_sky"
    assert response.json()["message"] == f"Added reference for '{sanitized_label}'"
    
    # Verify directory on disk uses underscores using Pathlib
    expected_dir = Path(temp_references_dir) / sanitized_label
    assert expected_dir.exists()
    assert expected_dir.is_dir()
    assert (expected_dir / "sky.png").exists()

def test_bulk_upload_with_spaces_in_folders(api_client, temp_references_dir):
    """Test bulk upload with folder names containing spaces."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        img1 = Image.new('RGB', (50, 50), color='gold')
        img1_buffer = io.BytesIO()
        img1.save(img1_buffer, format='JPEG')
        img1_buffer.seek(0)
        zf.writestr("golden retriever/dog1.jpg", img1_buffer.read())
        
        img2 = Image.new('RGB', (50, 50), color='brown')
        img2_buffer = io.BytesIO()
        img2.save(img2_buffer, format='JPEG')
        img2_buffer.seek(0)
        zf.writestr("german shepherd/dog2.jpg", img2_buffer.read())
        
    zip_buffer.seek(0)
    
    response = api_client.post(
        "/bulk_upload",
        files={"file": ("dogs_with_spaces.zip", zip_buffer, "application/zip")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "golden_retriever" in data["labels"]
    assert "german_shepherd" in data["labels"]
    
    # Verify disk uses underscores using Pathlib
    assert (Path(temp_references_dir) / "golden_retriever").exists()
    assert (Path(temp_references_dir) / "german_shepherd").exists()