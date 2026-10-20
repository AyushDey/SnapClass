import { useEffect, useState, useRef } from 'react';
import PropTypes from 'prop-types';

export default function Header({ backendStatus = 'checking', onRefreshCatalog, isRefreshing = false }) {
    const [isDark, setIsDark] = useState(() => {
        const saved = localStorage.getItem('snapclass-theme');
        if (saved) return saved === 'dark';
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    });

    // Bidirectional 1:1 drag state for theme toggle
    const [dragX, setDragX] = useState(null);
    const isDraggingRef = useRef(false);
    const startXRef = useRef(0);
    const startPosRef = useRef(0);

    useEffect(() => {
        const theme = isDark ? 'dark' : 'light';
        document.documentElement.dataset.theme = theme;
        localStorage.setItem('snapclass-theme', theme);
    }, [isDark]);

    // Pointer event handlers for draggable toggle
    const handleTogglePointerDown = (e) => {
        if (e.button !== 0) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        startXRef.current = e.clientX;
        startPosRef.current = isDark ? 34 : 0;
        isDraggingRef.current = false;
    };

    const handleTogglePointerMove = (e) => {
        if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
        const deltaX = e.clientX - startXRef.current;
        if (!isDraggingRef.current && Math.abs(deltaX) > 2) {
            isDraggingRef.current = true;
        }
        if (isDraggingRef.current) {
            const nextX = Math.max(0, Math.min(34, startPosRef.current + deltaX));
            setDragX(nextX);
        }
    };

    const handleTogglePointerUp = (e) => {
        if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
        e.currentTarget.releasePointerCapture(e.pointerId);

        if (isDraggingRef.current && dragX !== null) {
            // Drag commit: if thumb is past halfway (> 17px), snap to dark
            setIsDark(dragX > 17);
        } else {
            // Click or tap commit
            const rect = e.currentTarget.getBoundingClientRect();
            const clickOffset = e.clientX - rect.left;
            setIsDark(clickOffset >= 34);
        }

        isDraggingRef.current = false;
        setDragX(null);
    };

    const handleTogglePointerCancel = (e) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId);
        }
        isDraggingRef.current = false;
        setDragX(null);
    };

    // Current thumb transform: live during drag, snapped otherwise
    const thumbTransform = dragX !== null
        ? `translateX(${dragX}px)`
        : isDark
        ? 'translateX(34px)'
        : 'translateX(0px)';

    return (
        <header className="glass header-bar" role="banner">
            <div className="header-brand">
                <div className="app-icon" aria-hidden="true">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                        <circle cx="12" cy="13" r="4" />
                    </svg>
                </div>
                <div className="brand-text">
                    <div className="brand-title">
                        <span>SnapClass</span>
                    </div>
                    <span className="brand-subtitle">Few-Shot Visual Intelligence</span>
                </div>
            </div>

            <div className="header-controls">
                {/* Real-time backend status badge with instant hover tooltip */}
                <div 
                    className={`status-pill ${backendStatus}`} 
                    tabIndex={0}
                    role="status"
                    aria-label={`Backend status: ${backendStatus}`}
                >
                    <span className="status-dot" aria-hidden="true" />
                    <span className="status-label">
                        {backendStatus === 'connected' ? 'Connected' : backendStatus === 'checking' ? 'Connecting…' : 'Offline'}
                    </span>

                    {/* Instant hover tooltip detailing connection state */}
                    <div className="status-tooltip" role="tooltip">
                        <div className="status-tooltip-header">
                            <span className={`status-tooltip-indicator ${backendStatus}`} />
                            <span className="status-tooltip-title">
                                {backendStatus === 'connected' ? 'Backend Connected' : backendStatus === 'checking' ? 'Connecting to API' : 'Backend Disconnected'}
                            </span>
                        </div>
                        <span className="status-tooltip-desc">
                            {backendStatus === 'connected' 
                                ? 'Local API server is online and responding' 
                                : backendStatus === 'checking' 
                                ? 'Verifying connection to http://127.0.0.1:8000…' 
                                : 'Server unreachable at http://127.0.0.1:8000'}
                        </span>
                    </div>
                </div>

                {/* Refresh Catalog button */}
                {onRefreshCatalog && (
                    <button
                        className="btn-icon"
                        onClick={onRefreshCatalog}
                        disabled={isRefreshing || backendStatus !== 'connected'}
                        aria-label="Refresh reference embeddings"
                        title="Reload catalog references from disk"
                    >
                        <svg 
                            className={isRefreshing ? 'spin' : ''} 
                            width="16" 
                            height="16" 
                            viewBox="0 0 24 24" 
                            fill="none" 
                            stroke="currentColor" 
                            strokeWidth="2" 
                            strokeLinecap="round" 
                            strokeLinejoin="round"
                        >
                            <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
                        </svg>
                    </button>
                )}

                {/* Theme toggle segmented control with bidirectional dragging */}
                <div
                    className={`theme-toggle ${dragX !== null ? 'dragging' : ''}`}
                    role="radiogroup"
                    aria-label="Color theme selector"
                    tabIndex={0}
                    onPointerDown={handleTogglePointerDown}
                    onPointerMove={handleTogglePointerMove}
                    onPointerUp={handleTogglePointerUp}
                    onPointerCancel={handleTogglePointerCancel}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            setIsDark((d) => !d);
                        } else if (e.key === 'ArrowLeft') {
                            e.preventDefault();
                            setIsDark(false);
                        } else if (e.key === 'ArrowRight') {
                            e.preventDefault();
                            setIsDark(true);
                        }
                    }}
                >
                    <div 
                        className={`theme-toggle-slider ${dragX !== null ? 'is-dragging' : isDark ? 'dark' : 'light'}`} 
                        style={{
                            transform: thumbTransform,
                            transition: dragX !== null ? 'none' : undefined,
                        }}
                    />
                    <button
                        type="button"
                        className={`theme-option sun ${isDark ? '' : 'active'}`}
                        aria-label="Switch to Light mode"
                        tabIndex={-1}
                        onClick={(e) => {
                            e.stopPropagation();
                            setIsDark(false);
                        }}
                    >
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="5" />
                            <line x1="12" y1="1" x2="12" y2="3" />
                            <line x1="12" y1="21" x2="12" y2="23" />
                            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                            <line x1="1" y1="12" x2="3" y2="12" />
                            <line x1="21" y1="12" x2="23" y2="12" />
                            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                        </svg>
                    </button>
                    <button
                        type="button"
                        className={`theme-option moon ${isDark ? 'active' : ''}`}
                        aria-label="Switch to Dark mode"
                        tabIndex={-1}
                        onClick={(e) => {
                            e.stopPropagation();
                            setIsDark(true);
                        }}
                    >
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                        </svg>
                    </button>
                </div>
            </div>
        </header>
    );
}

Header.propTypes = {
    backendStatus: PropTypes.oneOf(['checking', 'connected', 'error']),
    onRefreshCatalog: PropTypes.func,
    isRefreshing: PropTypes.bool,
};
