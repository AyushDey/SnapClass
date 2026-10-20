import { useRef, useState } from 'react';
import PropTypes from 'prop-types';
import CameraBox from './CameraBox';

export default function ClassifyPanel({
    videoRef,
    cameraActive,
    capturedImage,
    uploadedPreview,
    onStartCamera,
    onStopCamera,
    onCapture,
    onUpload,
    onClearImage,
    onClassify,
    onBulkUpload,
    isClassifying,
    hasImage,
}) {
    const [mode, setMode] = useState('camera'); // 'camera' | 'upload' | 'dataset'
    const fileInputRef = useRef(null);
    const bulkInputRef = useRef(null);

    const handleFileSelect = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            onUpload(file);
        }
        e.target.value = '';
    };

    const handleBulkSelect = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            onBulkUpload(file);
        }
        e.target.value = '';
    };

    return (
        <section className="glass panel classify-panel" aria-label="Image Acquisition and Classification">
            {/* Header & Segmented Mode Switcher */}
            <div className="panel-header">
                <div>
                    <h2 className="panel-title">Visual Capture</h2>
                    <p className="panel-subtitle">Capture from camera or select an image file</p>
                </div>
                
                <div className="segmented-control" role="tablist" aria-label="Input mode">
                    <button
                        type="button"
                        role="tab"
                        aria-selected={mode === 'camera'}
                        className={`segmented-btn ${mode === 'camera' ? 'active' : ''}`}
                        onClick={() => {
                            setMode('camera');
                        }}
                    >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                            <circle cx="12" cy="13" r="4" />
                        </svg>
                        <span>Camera</span>
                    </button>
                    <button
                        type="button"
                        role="tab"
                        aria-selected={mode === 'upload'}
                        className={`segmented-btn ${mode === 'upload' ? 'active' : ''}`}
                        onClick={() => {
                            if (cameraActive && onStopCamera) onStopCamera();
                            setMode('upload');
                        }}
                    >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                        <span>Upload</span>
                    </button>
                    <button
                        type="button"
                        role="tab"
                        aria-selected={mode === 'dataset'}
                        className={`segmented-btn ${mode === 'dataset' ? 'active' : ''}`}
                        onClick={() => {
                            if (cameraActive && onStopCamera) onStopCamera();
                            setMode('dataset');
                        }}
                    >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                            <line x1="12" y1="22.08" x2="12" y2="12" />
                        </svg>
                        <span>Catalog</span>
                    </button>
                </div>
            </div>

            {/* Viewfinder / Preview Area */}
            {mode !== 'dataset' ? (
                <>
                    <CameraBox
                        videoRef={videoRef}
                        isActive={cameraActive}
                        capturedImage={capturedImage}
                        uploadedPreview={uploadedPreview}
                        isClassifying={isClassifying}
                        onClear={onClearImage}
                        onDropFile={onUpload}
                    />

                    {/* Mode Specific Controls */}
                    <div className="input-actions-strip">
                        {mode === 'camera' && (
                            cameraActive ? (
                                <div className="camera-live-controls">
                                    <button
                                        type="button"
                                        className="btn btn-secondary btn-sm"
                                        onClick={onStopCamera}
                                    >
                                        Cancel
                                    </button>
                                    
                                    {/* Apple Camera Shutter Button */}
                                    <button
                                        type="button"
                                        className="shutter-button"
                                        onClick={onCapture}
                                        aria-label="Capture snapshot"
                                    >
                                        <span className="shutter-inner" />
                                    </button>

                                    <div style={{ width: '60px' }} />
                                </div>
                            ) : (
                                <button
                                    type="button"
                                    className="btn btn-secondary w-full"
                                    onClick={onStartCamera}
                                >
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                                        <circle cx="12" cy="13" r="4" />
                                    </svg>
                                    <span>Open Camera</span>
                                </button>
                            )
                        )}

                        {mode === 'upload' && (
                            <div className="upload-cta-box">
                                <button
                                    type="button"
                                    className="btn btn-secondary w-full"
                                    onClick={() => fileInputRef.current?.click()}
                                >
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                        <polyline points="17 8 12 3 7 8" />
                                        <line x1="12" y1="3" x2="12" y2="15" />
                                    </svg>
                                    <span>Browse Image File</span>
                                </button>
                                <span className="format-caption">Supported formats: JPEG, PNG, WebP, BMP</span>
                            </div>
                        )}

                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="image/jpeg,image/png,image/webp,image/bmp"
                            className="file-input-hidden"
                            onChange={handleFileSelect}
                        />
                    </div>

                    {/* Primary Hero Classify Button */}
                    <button
                        type="button"
                        className="btn btn-primary btn-hero"
                        onClick={onClassify}
                        disabled={!hasImage || isClassifying}
                    >
                        {isClassifying ? (
                            <>
                                <span className="spinner-dot" />
                                <span>Computing Embeddings…</span>
                            </>
                        ) : (
                            <>
                                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                                    <circle cx="12" cy="12" r="10" />
                                    <line x1="22" y1="12" x2="18" y2="12" />
                                    <line x1="6" y1="12" x2="2" y2="12" />
                                    <line x1="12" y1="6" x2="12" y2="2" />
                                    <line x1="12" y1="22" x2="12" y2="18" />
                                </svg>
                                <span>Run Classification</span>
                            </>
                        )}
                    </button>
                </>
            ) : (
                /* Catalog / Bulk Archive Manager Mode */
                <div className="dataset-card">
                    <div className="dataset-icon">
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                            <line x1="12" y1="22.08" x2="12" y2="12" />
                        </svg>
                    </div>

                    <div className="dataset-info">
                        <h3>Bulk Reference Ingestion</h3>
                        <p>Upload a compressed archive (.zip, .tar, .tar.gz) with nested class references to enrich the database in real time.</p>
                        <div className="folder-structure-pill">
                            <code>Category / ClassName / images.*</code>
                        </div>
                    </div>

                    <button
                        type="button"
                        className="btn btn-primary w-full"
                        onClick={() => bulkInputRef.current?.click()}
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                        <span>Select Archive (.zip / .tar)</span>
                    </button>

                    <input
                        ref={bulkInputRef}
                        type="file"
                        accept=".zip,.tar,.tar.gz,.tgz,.tar.bz2"
                        className="file-input-hidden"
                        onChange={handleBulkSelect}
                    />
                </div>
            )}
        </section>
    );
}

ClassifyPanel.propTypes = {
    videoRef: PropTypes.shape({ current: PropTypes.object }),
    cameraActive: PropTypes.bool.isRequired,
    capturedImage: PropTypes.string,
    uploadedPreview: PropTypes.string,
    onStartCamera: PropTypes.func.isRequired,
    onStopCamera: PropTypes.func,
    onCapture: PropTypes.func.isRequired,
    onUpload: PropTypes.func.isRequired,
    onClearImage: PropTypes.func,
    onClassify: PropTypes.func.isRequired,
    onBulkUpload: PropTypes.func.isRequired,
    isClassifying: PropTypes.bool.isRequired,
    hasImage: PropTypes.bool.isRequired,
};
