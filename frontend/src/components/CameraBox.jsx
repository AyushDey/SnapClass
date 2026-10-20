import { useState } from 'react';
import PropTypes from 'prop-types';

export default function CameraBox({
    videoRef,
    isActive,
    capturedImage,
    uploadedPreview,
    isClassifying = false,
    onClear,
    onDropFile,
}) {
    const [isDragOver, setIsDragOver] = useState(false);
    const showVideo = isActive && !capturedImage;
    const previewSrc = capturedImage || uploadedPreview;

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!isDragOver) setIsDragOver(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragOver(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragOver(false);
        const files = e.dataTransfer?.files;
        if (files && files.length > 0 && onDropFile) {
            onDropFile(files[0]);
        }
    };

    return (
        <div
            className={`camera-box ${isActive ? 'active' : ''} ${isDragOver ? 'drag-over' : ''} ${previewSrc ? 'has-preview' : ''}`}
            onDragOver={handleDragOver}
            onDragEnter={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            aria-label="Camera viewfinder and image drop zone"
        >
            {/* Viewfinder Reticles (Apple Camera UI) */}
            <div className="viewfinder-reticle top-left" aria-hidden="true" />
            <div className="viewfinder-reticle top-right" aria-hidden="true" />
            <div className="viewfinder-reticle bottom-left" aria-hidden="true" />
            <div className="viewfinder-reticle bottom-right" aria-hidden="true" />

            {/* Live Camera Feed */}
            <video
                ref={videoRef}
                playsInline
                muted
                autoPlay
                className={`camera-video ${showVideo ? 'visible' : ''}`}
            />

            {/* Captured or Uploaded Preview */}
            {previewSrc && !showVideo && (
                <div className="preview-wrapper">
                    <img src={previewSrc} alt="Input candidate preview" className="preview-image" />
                    {isClassifying && (
                        <div className="scanner-beam" aria-label="Analyzing visual features" />
                    )}
                </div>
            )}

            {/* Live Camera Badge */}
            {showVideo && (
                <div className="camera-live-badge">
                    <span className="live-dot" />
                    <span>LIVE</span>
                </div>
            )}

            {/* Retake / Clear Action Button */}
            {previewSrc && !isClassifying && onClear && (
                <button
                    type="button"
                    className="preview-clear-btn"
                    onClick={onClear}
                    title="Remove image / Retake"
                    aria-label="Remove image and retake"
                >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                </button>
            )}

            {/* Empty State Placeholder */}
            {!previewSrc && !showVideo && (
                <div className="camera-placeholder">
                    <div className="placeholder-icon-wrap">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                            <circle cx="12" cy="13" r="4" />
                        </svg>
                    </div>
                    <div className="placeholder-text">
                        <strong>Drop image here</strong>
                        <span>or use camera / upload below</span>
                    </div>
                </div>
            )}

            {/* Drag Over Overlay */}
            {isDragOver && (
                <div className="drag-overlay">
                    <div className="drag-icon-wrap">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                    </div>
                    <span>Release image to inspect</span>
                </div>
            )}
        </div>
    );
}

CameraBox.propTypes = {
    videoRef: PropTypes.shape({ current: PropTypes.object }),
    isActive: PropTypes.bool.isRequired,
    capturedImage: PropTypes.string,
    uploadedPreview: PropTypes.string,
    isClassifying: PropTypes.bool,
    onClear: PropTypes.func,
    onDropFile: PropTypes.func,
};
