import { useState, useCallback, useEffect } from 'react';
import Header from './components/Header';
import ClassifyPanel from './components/ClassifyPanel';
import ResultPanel from './components/ResultPanel';
import { useCamera } from './hooks/useCamera';
import { classifyImage, bulkUpload, refreshReferences, checkHealth } from './api/snapclass';

export default function App() {
    const {
        videoRef,
        isActive: cameraActive,
        capturedImage,
        startCamera,
        stopCamera,
        captureSnapshot,
        clearCapture,
    } = useCamera();

    const [uploadedPreview, setUploadedPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [isClassifying, setIsClassifying] = useState(false);
    const [loading, setLoading] = useState(null);
    const [toast, setToast] = useState(null);
    const [pendingFile, setPendingFile] = useState(null);
    const [backendStatus, setBackendStatus] = useState('checking'); // 'checking' | 'connected' | 'error'
    const [isRefreshing, setIsRefreshing] = useState(false);

    // Dynamic Island / HUD toast helper
    const showToast = useCallback((message, type = 'success', title = null) => {
        setToast({ message, type, title });
        setTimeout(() => setToast(null), 3800);
    }, []);

    // Initial and periodic backend health check
    const verifyBackend = useCallback(async () => {
        try {
            await checkHealth();
            setBackendStatus('connected');
        } catch {
            setBackendStatus('error');
        }
    }, []);

    useEffect(() => {
        verifyBackend();
        const interval = setInterval(verifyBackend, 15000);
        return () => clearInterval(interval);
    }, [verifyBackend]);

    const handleClearImage = useCallback(() => {
        setPendingFile(null);
        setUploadedPreview(null);
        clearCapture();
        setResult(null);
    }, [clearCapture]);

    // Keyboard navigation (Esc to reset preview)
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') {
                if (pendingFile) {
                    handleClearImage();
                }
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [pendingFile, handleClearImage]);

    const handleStartCamera = useCallback(async () => {
        try {
            setUploadedPreview(null);
            setPendingFile(null);
            setResult(null);
            await startCamera();
        } catch {
            showToast('Unable to access camera. Please verify camera permissions.', 'error', 'Permission Denied');
        }
    }, [startCamera, showToast]);

    const handleCapture = useCallback(async () => {
        const file = await captureSnapshot();
        if (file) {
            setPendingFile(file);
            setResult(null);
        }
    }, [captureSnapshot]);

    const handleUpload = useCallback((file) => {
        setPendingFile(file);
        clearCapture();
        if (cameraActive) stopCamera();
        setResult(null);
        setUploadedPreview(URL.createObjectURL(file));
    }, [clearCapture, cameraActive, stopCamera]);

    const handleClassify = useCallback(async () => {
        if (!pendingFile) return;
        setIsClassifying(true);
        setResult(null);

        try {
            const data = await classifyImage(pendingFile);
            setResult(data);
            showToast(
                `Recognized as ${data.class.replace(/_/g, ' ')} with ${(data.confidence * 100).toFixed(1)}% confidence`,
                'success',
                'Classification Succeeded'
            );
        } catch (err) {
            showToast(err.message || 'Classification request failed', 'error', 'Inference Error');
        } finally {
            setIsClassifying(false);
        }
    }, [pendingFile, showToast]);

    const handleBulkUpload = useCallback(async (file) => {
        setLoading('Ingesting catalog archive & computing embeddings…');
        try {
            const data = await bulkUpload(file);
            showToast(
                `Successfully indexed ${data.total_images} images across ${Object.keys(data.labels || {}).length} classes`,
                'success',
                'Catalog Updated'
            );
            verifyBackend();
        } catch (err) {
            showToast(err.message || 'Bulk archive ingestion failed', 'error', 'Upload Failed');
        } finally {
            setLoading(null);
        }
    }, [showToast, verifyBackend]);

    const handleRefreshCatalog = useCallback(async () => {
        setIsRefreshing(true);
        try {
            const res = await refreshReferences();
            const totalClasses = res.classes?.length || 0;
            const added = res.added || 0;
            const removed = res.removed || 0;

            if (added > 0 || removed > 0) {
                const changes = [];
                if (added > 0) changes.push(`${added} new image${added > 1 ? 's' : ''} added`);
                if (removed > 0) changes.push(`${removed} stale item${removed > 1 ? 's' : ''} removed`);
                showToast(
                    `Catalog updated: ${changes.join(', ')} (${totalClasses} classes total)`,
                    'success',
                    'Catalog Updated'
                );
            } else {
                showToast(
                    `Catalog is up to date (${totalClasses} classes indexed, 0 new items)`,
                    'info',
                    'Catalog Up to Date'
                );
            }
        } catch (err) {
            showToast(err.message || 'Failed to refresh embeddings', 'error', 'Refresh Failed');
        } finally {
            setIsRefreshing(false);
        }
    }, [showToast]);

    return (
        <div className="app-container">
            <Header
                backendStatus={backendStatus}
                onRefreshCatalog={handleRefreshCatalog}
                isRefreshing={isRefreshing}
            />

            <main className="main-grid">
                <ClassifyPanel
                    videoRef={videoRef}
                    cameraActive={cameraActive}
                    capturedImage={capturedImage}
                    uploadedPreview={uploadedPreview}
                    onStartCamera={handleStartCamera}
                    onStopCamera={stopCamera}
                    onCapture={handleCapture}
                    onUpload={handleUpload}
                    onClearImage={handleClearImage}
                    onClassify={handleClassify}
                    onBulkUpload={handleBulkUpload}
                    isClassifying={isClassifying}
                    hasImage={!!pendingFile}
                />
                
                <ResultPanel
                    result={result}
                    isClassifying={isClassifying}
                />
            </main>

            {/* Apple Style Translucent Modal Loader */}
            {loading && (
                <div className="loading-scrim" role="dialog" aria-modal="true" aria-label="Processing archive">
                    <div className="glass loading-card">
                        <div className="apple-spinner" aria-hidden="true" />
                        <div className="loading-meta">
                            <h4>Processing Archive</h4>
                            <p>{loading}</p>
                        </div>
                    </div>
                </div>
            )}

            {/* Apple Dynamic Island / HUD Notification */}
            {toast && (
                <div className={`apple-hud-toast ${toast.type}`} role="status">
                    <div className="hud-icon">
                        {toast.type === 'success' ? (
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="20 6 9 17 4 12" />
                            </svg>
                        ) : toast.type === 'info' ? (
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="10" />
                                <line x1="12" y1="16" x2="12" y2="12" />
                                <line x1="12" y1="8" x2="12.01" y2="8" />
                            </svg>
                        ) : (
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="10" />
                                <line x1="12" y1="8" x2="12" y2="12" />
                                <line x1="12" y1="16" x2="12.01" y2="16" />
                            </svg>
                        )}
                    </div>
                    <div className="hud-content">
                        {toast.title && <div className="hud-title">{toast.title}</div>}
                        <div className="hud-message">{toast.message}</div>
                    </div>
                </div>
            )}
        </div>
    );
}
