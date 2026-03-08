import { useRef } from 'react';
import PropTypes from 'prop-types';
import CameraBox from './CameraBox';

export default function ClassifyPanel({
    videoRef,
    cameraActive,
    capturedImage,
    uploadedPreview,
    onStartCamera,
    onCapture,
    onUpload,
    onClassify,
    onBulkUpload,
    isClassifying,
    hasImage,
}) {
    const fileInputRef = useRef(null);
    const bulkInputRef = useRef(null);

    const handleFileSelect = (e) => {
        const file = e.target.files?.[0];
        if (file) onUpload(file);
        e.target.value = '';
    };

    const handleBulkSelect = (e) => {
        const file = e.target.files?.[0];
        if (file) onBulkUpload(file);
        e.target.value = '';
    };

    return (
        <section className="glass panel">
            <div className="panel-title">Classify</div>

            <CameraBox
                videoRef={videoRef}
                isActive={cameraActive}
                capturedImage={capturedImage}
                uploadedPreview={uploadedPreview}
            />

            <div className="camera-actions">
                {cameraActive ? (
                    <button className="btn btn-primary" onClick={onCapture}>
                        Capture
                    </button>
                ) : (
                    <button className="btn btn-secondary" onClick={onStartCamera}>
                        Open Camera
                    </button>
                )}

                <button
                    className="btn btn-secondary"
                    onClick={() => fileInputRef.current?.click()}
                >
                    Upload Image
                </button>

                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp,image/bmp"
                    className="file-input-hidden"
                    onChange={handleFileSelect}
                />
            </div>

            <button
                className="btn btn-primary"
                onClick={onClassify}
                disabled={!hasImage || isClassifying}
                style={{ width: '100%', padding: '12px', fontSize: '0.9375rem' }}
            >
                {isClassifying ? 'Classifying…' : 'Classify'}
            </button>

            <div className="bulk-section">
                <button
                    className="btn btn-secondary"
                    onClick={() => bulkInputRef.current?.click()}
                    style={{ width: '100%' }}
                >
                    Bulk Upload
                </button>
                <input
                    ref={bulkInputRef}
                    type="file"
                    accept=".zip,.tar,.tar.gz,.tgz,.tar.bz2"
                    className="file-input-hidden"
                    onChange={handleBulkSelect}
                />
            </div>
        </section>
    );
}

ClassifyPanel.propTypes = {
    videoRef: PropTypes.shape({ current: PropTypes.object }),
    cameraActive: PropTypes.bool.isRequired,
    capturedImage: PropTypes.string,
    uploadedPreview: PropTypes.string,
    onStartCamera: PropTypes.func.isRequired,
    onCapture: PropTypes.func.isRequired,
    onUpload: PropTypes.func.isRequired,
    onClassify: PropTypes.func.isRequired,
    onBulkUpload: PropTypes.func.isRequired,
    isClassifying: PropTypes.bool.isRequired,
    hasImage: PropTypes.bool.isRequired
};
