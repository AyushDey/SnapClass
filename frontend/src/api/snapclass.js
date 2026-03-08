const API_BASE = 'http://localhost:8000';

/**
 * Classify an image file.
 * @param {File|Blob} file
 * @returns {Promise<Object>} classification result
 */
export async function classifyImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/classify`, {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Classification failed');
    }

    return res.json();
}

/**
 * Upload an archive for bulk reference import.
 * @param {File} file .zip / .tar / .tar.gz
 * @returns {Promise<Object>}
 */
export async function bulkUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/bulk_upload`, {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Bulk upload failed');
    }

    return res.json();
}

/**
 * Refresh the reference embeddings.
 * @returns {Promise<Object>}
 */
export async function refreshReferences() {
    const res = await fetch(`${API_BASE}/refresh`, { method: 'POST' });

    if (!res.ok) {
        throw new Error('Refresh failed');
    }

    return res.json();
}
