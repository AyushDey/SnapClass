
import pytest
import io
from PIL import Image

def test_classify_type_mismatch_regression(api_client, temp_references_dir, mock_image):
    """
    Regression test for: RuntimeError: expected m1 and m2 to have same dtype but got: float != double
    
    This happens because:
    1. References are loaded from database as lists of floats.
    2. numpy.array(list) defaults to float64 (double).
    3. torch.tensor(numpy_array) inherits float64.
    4. ResNet output is float32.
    5. torch.mm(float32, float64) fails.
    """
    
    # 1. Add a reference (creates proper referencing in DB)
    # We can use the existing /add_reference endpoint or manually setup.
    # Using endpoint is safer as it exercises the full flow.
    img_byte_arr = io.BytesIO()
    mock_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    response = api_client.post(
        "/add_reference",
        data={"label": "test_label"},
        files={"file": ("reference.png", img_byte_arr, "image/png")}
    )
    assert response.status_code == 200
    
    # 2. Trigger classification
    # This invokes classifier.classify -> torch.mm
    img_byte_arr.seek(0)
    response = api_client.post(
        "/classify",
        files={"file": ("query.png", img_byte_arr, "image/png")}
    )
    
    # If the bug exists, this will likely return 500 or raise an error in the app
    assert response.status_code == 200
    data = response.json()
    assert data["class"] == "test_label"
