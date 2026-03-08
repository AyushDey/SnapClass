import { useEffect, useState } from 'react';

export default function Header() {
    const [isDark, setIsDark] = useState(() => {
        return (localStorage.getItem('snapclass-theme') || 'dark') === 'dark';
    });

    useEffect(() => {
        const theme = isDark ? 'dark' : 'light';
        document.documentElement.dataset.theme = theme;
        localStorage.setItem('snapclass-theme', theme);
    }, [isDark]);

    return (
        <header
            className="glass panel"
            style={{
                flexDirection: 'row',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '14px 24px',
            }}
        >
            <h1 style={{ fontSize: '1.125rem', fontWeight: 600, letterSpacing: '-0.02em' }}>
                SnapClass{' '}
                <span
                    style={{
                        fontWeight: 400,
                        color: 'var(--text-secondary)',
                        marginLeft: '8px',
                        fontSize: '1rem',
                    }}
                >
                    AI Image Recognition
                </span>
            </h1>

            {/* Theme toggle — segmented pill with sun & moon */}
            <div
                className="theme-toggle"
                role="radiogroup"
                aria-label="Color theme"
                onClick={() => setIsDark((d) => !d)}
                tabIndex={0}
                onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setIsDark((d) => !d);
                    }
                }}
            >
                <div className={`theme-toggle-slider ${isDark ? 'dark' : 'light'}`} />
                <button
                    className={`theme-option sun ${isDark ? '' : 'active'}`}
                    aria-label="Light mode"
                    tabIndex={-1}
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
                    className={`theme-option moon ${isDark ? 'active' : ''}`}
                    aria-label="Dark mode"
                    tabIndex={-1}
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                    </svg>
                </button>
            </div>
        </header>
    );
}
