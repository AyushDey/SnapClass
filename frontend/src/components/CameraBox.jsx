import PropTypes from 'prop-types';

export default function CameraBox({ videoRef, isActive, capturedImage, uploadedPreview }) {
    const showVideo = isActive && !capturedImage;
    const previewSrc = capturedImage || uploadedPreview;

    return (
        <div className={`camera-box ${isActive ? 'active' : ''}`}>
            {isActive && !capturedImage && <div className="camera-dot" />}

            <video
                ref={videoRef}
                playsInline
                muted
                style={{ display: showVideo ? 'block' : 'none' }}
            />

            {previewSrc && !showVideo && (
                <img src={previewSrc} alt="Preview" />
            )}

            {!previewSrc && !showVideo && (
                <div className="camera-placeholder">
                    <div className="icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                            <circle cx="12" cy="13" r="4" />
                        </svg>
                    </div>
                    <span>Open camera or upload an image</span>
                </div>
            )}
        </div>
    );
}

CameraBox.propTypes = {
    videoRef: PropTypes.shape({ current: PropTypes.object }),
    isActive: PropTypes.bool.isRequired,
    capturedImage: PropTypes.string,
    uploadedPreview: PropTypes.string
};
