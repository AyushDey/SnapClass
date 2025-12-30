import os
import io
from PIL import Image

def test_root_endpoint(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "active"

def test_add_reference_flow(api_client, temp_references_dir):
    # 1. Create a dummy image
    img = Image.new('RGB', (100, 100), color='blue')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # 2. Upload
    response = api_client.post(
        "/add_reference",
        data={"label": "blue_box"},
        files={"file": ("blue.png", img_byte_arr, "image/png")}
    )
    
    assert response.status_code == 200
    assert "Added reference" in response.json()["message"]
    
    # 3. Verify file on disk
    label_path = os.path.join(temp_references_dir, "blue_box")
    assert os.path.exists(label_path)
    assert len(os.listdir(label_path)) == 1

def test_classify_endpoint_unknown(api_client, mock_image):
    # Determine behavior when no refs are present (conftest sets temp dir empty)
    img_byte_arr = io.BytesIO()
    mock_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    response = api_client.post(
        "/classify",
        files={"file": ("test.png", img_byte_arr, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["class"] == "Unknown"

def test_invalid_file_type(api_client):
    response = api_client.post(
        "/classify",
        files={"file": ("test.txt", b"some text", "text/plain")}
    )
    # The API raises 400 for invalid extension
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_refresh_endpoint(api_client):
    response = api_client.post("/refresh")
    assert response.status_code == 200
    assert "References reloaded" in response.json()["message"]
