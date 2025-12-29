import requests
import os
from PIL import Image, ImageDraw

def create_color_image(color, filename):
    img = Image.new('RGB', (100, 100), color=color)
    img.save(filename)
    return filename

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    # ensure clean state
    # (Manual step: assume server started empty or handles overwrite)

    print("1. Creating dummy reference images...")
    red_img = create_color_image("red", "red.jpg")
    blue_img = create_color_image("blue", "blue.jpg")
    green_img = create_color_image("green", "green.jpg")
    
    # Add references via API
    print("2. Uploading references...")
    with open(red_img, "rb") as f:
        requests.post(f"{base_url}/add_reference", data={"label": "color_red"}, files={"file": f})
    with open(blue_img, "rb") as f:
        requests.post(f"{base_url}/add_reference", data={"label": "color_blue"}, files={"file": f})
        
    # Test Classification
    print("3. Testing Classification...")
    
    # Should match red
    with open(red_img, "rb") as f:
        res = requests.post(f"{base_url}/classify", files={"file": f})
        print(f"Red Image: {res.json()}")
        
    # Should match blue
    with open(blue_img, "rb") as f:
        res = requests.post(f"{base_url}/classify", files={"file": f})
        print(f"Blue Image: {res.json()}")
        
    # Should be Unknown (Green)
    with open(green_img, "rb") as f:
        res = requests.post(f"{base_url}/classify", files={"file": f})
        print(f"Green Image (Should be Unknown or low conf): {res.json()}")

    # Cleanup
    for f in [red_img, blue_img, green_img]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    try:
        test_api()
    except Exception as e:
        print(f"Test failed (is server running?): {e}")
