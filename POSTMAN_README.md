# SnapClass API - Postman Testing Guide

This guide will help you test the endpoints of the SnapClass API using Postman.

## 1. Setup

1.  **Install Postman**: If you haven't already, download and install [Postman](https://www.postman.com/downloads/).
2.  **Ensure API is Running**: Make sure your FastAPI server is running.
    ```bash
    uvicorn main:app --reload
    # Or with Docker:
    docker run -p 8000:8000 -v $(pwd)/references:/app/references -v $(pwd)/chroma_db:/app/chroma_db snapclass
    ```
    Default URL: `http://127.0.0.1:8000`

## 2. Import Collection

1.  Open Postman.
2.  Click the **Import** button (top left).
3.  Drag and drop the `postman_collection.json` file from this project into the import window, or select it via the file browser.
4.  Click **Import**.

## 3. Using the Collection

The collection "SnapClass API" will appear in your workspace. It uses a collection variable `base_url` set to `http://127.0.0.1:8000` by default.

### Available Endpoints

#### **GET / (Check Health)**
- Select "Check Health" and click **Send**.
- You should receive a status message confirming the API is running.

#### **POST /classify**
- Select "Classify Image".
- Go to the **Body** tab.
- You will see a `file` key.
- **Action Required**: Hover over the Value column for `file` and ensure the type is set to **File** (dropdown menu), then click **Select Files** to choose an image from your computer to test.
- Click **Send** to see the classification result.

#### **POST /add_reference**
- Select "Add Reference Image".
- Go to the **Body** tab.
- **Action Required**:
    - Update the `label` value (default is "binoculars") to the class you want to add.
    - For the `file` key, ensure type is **File**, then select an image you want to use as a reference.
- Click **Send**. Returns a success message.

#### **POST /refresh**
- Select "Refresh References".
- Click **Send**.
- This reloads all images from the `references/` directory into the classifier's memory. Call this if you manually added files to the folder or want to ensure the state is fresh.

---
**Tip**: If your server is running on a different port, click on the Collection name ("SnapClass API"), go to the **Variables** tab, and update the `base_url` value.
