import { useState, useRef, useCallback } from 'react';

/**
 * Custom hook to manage webcam stream and snapshot capture.
 */
export function useCamera() {
    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const [isActive, setIsActive] = useState(false);
    const [capturedImage, setCapturedImage] = useState(null);

    const startCamera = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: 640, height: 480 },
            });
            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }

            setCapturedImage(null);
            setIsActive(true);
        } catch (err) {
            console.error('Camera access denied:', err);
            throw err;
        }
    }, []);

    const stopCamera = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((t) => t.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        setIsActive(false);
    }, []);

    /**
     * Capture current video frame as a Blob (JPEG).
     * Returns both a data-URL for preview and a File for upload.
     */
    const captureSnapshot = useCallback(() => {
        if (!videoRef.current) return null;

        const video = videoRef.current;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
        setCapturedImage(dataUrl);

        // Stop the camera after capture
        stopCamera();

        // Convert to Blob/File for uploading
        return new Promise((resolve) => {
            canvas.toBlob(
                (blob) => {
                    const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' });
                    resolve(file);
                },
                'image/jpeg',
                0.92
            );
        });
    }, [stopCamera]);

    const clearCapture = useCallback(() => {
        setCapturedImage(null);
    }, []);

    return {
        videoRef,
        isActive,
        capturedImage,
        startCamera,
        stopCamera,
        captureSnapshot,
        clearCapture,
    };
}
