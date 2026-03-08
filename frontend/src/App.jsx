import { useState, useCallback } from 'react';
import Header from './components/Header';
import ClassifyPanel from './components/ClassifyPanel';
import ResultPanel from './components/ResultPanel';
import { useCamera } from './hooks/useCamera';
import { classifyImage, bulkUpload } from './api/snapclass';

export default function App() {
  const {
    videoRef,
    isActive: cameraActive,
    capturedImage,
    startCamera,
    captureSnapshot,
    clearCapture,
  } = useCamera();

  const [uploadedPreview, setUploadedPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [loading, setLoading] = useState(null);
  const [toast, setToast] = useState(null);
  const [pendingFile, setPendingFile] = useState(null);

  const showToast = useCallback((message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  }, []);

  const handleStartCamera = useCallback(async () => {
    try {
      setUploadedPreview(null);
      setPendingFile(null);
      await startCamera();
    } catch {
      showToast('Camera access denied.', 'error');
    }
  }, [startCamera, showToast]);

  const handleCapture = useCallback(async () => {
    const file = await captureSnapshot();
    if (file) setPendingFile(file);
  }, [captureSnapshot]);

  const handleUpload = useCallback((file) => {
    setPendingFile(file);
    clearCapture();
    setUploadedPreview(URL.createObjectURL(file));
  }, [clearCapture]);

  const handleClassify = useCallback(async () => {
    if (!pendingFile) return;
    setIsClassifying(true);
    setResult(null);

    try {
      const data = await classifyImage(pendingFile);
      setResult(data);
    } catch (err) {
      showToast(err.message || 'Classification failed', 'error');
    } finally {
      setIsClassifying(false);
    }
  }, [pendingFile, showToast]);

  const handleBulkUpload = useCallback(async (file) => {
    setLoading('Processing archive…');
    try {
      const data = await bulkUpload(file);
      showToast(
        `Upload complete — ${data.total_images} images, ${Object.keys(data.labels || {}).length} labels`
      );
    } catch (err) {
      showToast(err.message || 'Upload failed', 'error');
    } finally {
      setLoading(null);
    }
  }, [showToast]);

  return (
    <div className="app-container">
      <Header />

      <div className="main-grid">
        <ClassifyPanel
          videoRef={videoRef}
          cameraActive={cameraActive}
          capturedImage={capturedImage}
          uploadedPreview={uploadedPreview}
          onStartCamera={handleStartCamera}
          onCapture={handleCapture}
          onUpload={handleUpload}
          onClassify={handleClassify}
          onBulkUpload={handleBulkUpload}
          isClassifying={isClassifying}
          hasImage={!!pendingFile}
        />
        <ResultPanel result={result} />
      </div>

      {loading && (
        <div className="loading-overlay">
          <div className="spinner" />
          <div className="loading-text">{loading}</div>
        </div>
      )}

      {toast && (
        <div className={`toast ${toast.type}`}>{toast.message}</div>
      )}
    </div>
  );
}
