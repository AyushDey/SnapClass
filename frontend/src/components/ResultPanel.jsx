import { useState } from 'react';
import PropTypes from 'prop-types';
import MatchCard from './MatchCard';
import { API_BASE } from '../api/snapclass';

export default function ResultPanel({ result, isClassifying = false }) {
    const [accordionOpen, setAccordionOpen] = useState(true);

    // Construct image URL using API_BASE
    const getImageUrl = (path) => {
        if (!path) return null;
        if (path.startsWith('http://') || path.startsWith('https://')) return path;
        return `${API_BASE}${path}`;
    };

    if (isClassifying) {
        return (
            <section className="glass panel result-panel" aria-live="polite" aria-busy="true">
                <div className="panel-header">
                    <div>
                        <h2 className="panel-title">Classification Results</h2>
                        <p className="panel-subtitle">Computing visual feature embeddings…</p>
                    </div>
                </div>

                <div className="skeleton-hero-card">
                    <div className="skeleton-pulse skeleton-thumb" />
                    <div className="skeleton-details">
                        <div className="skeleton-pulse skeleton-tag" />
                        <div className="skeleton-pulse skeleton-title" />
                        <div className="skeleton-pulse skeleton-bar" />
                    </div>
                </div>

                <div className="skeleton-list">
                    <div className="skeleton-pulse skeleton-row" />
                    <div className="skeleton-pulse skeleton-row" />
                </div>
            </section>
        );
    }

    if (!result) {
        return (
            <section className="glass panel result-panel" aria-label="Recognition Results">
                <div className="panel-header">
                    <div>
                        <h2 className="panel-title">Classification Results</h2>
                        <p className="panel-subtitle">Matches will display here after capture</p>
                    </div>
                </div>

                <div className="result-placeholder">
                    <div className="placeholder-scan-glyph" aria-hidden="true">
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="9" />
                            <path d="M12 3v3" />
                            <path d="M12 18v3" />
                            <path d="M3 12h3" />
                            <path d="M18 12h3" />
                            <circle cx="12" cy="12" r="3" />
                        </svg>
                    </div>
                    <div className="placeholder-text-block">
                        <h3>No Image Loaded</h3>
                        <p>Capture a photo or upload an image file to compare against reference classes.</p>
                    </div>
                </div>
            </section>
        );
    }

    const topMatch = {
        label: result.class,
        class: result.class,
        confidence: result.confidence,
        category: result.category_name,
        category_name: result.category_name,
        image_url: getImageUrl(result.image_path),
    };

    const otherMatches = (result.matches || []).map((m) => ({
        label: m.class,
        class: m.class,
        confidence: m.score,
        score: m.score,
        category: m.category_name,
        category_name: m.category_name,
        image_url: getImageUrl(m.image_path),
    }));

    return (
        <section className="glass panel result-panel" aria-live="polite">
            <div className="panel-header">
                <div>
                    <h2 className="panel-title">Classification Results</h2>
                    <p className="panel-subtitle">Ranked matches from visual catalog</p>
                </div>
                <span className="results-count-pill">
                    {(otherMatches.length + 1)} Candidates
                </span>
            </div>

            <MatchCard match={topMatch} isTopMatch rank={1} />

            {otherMatches.length > 0 && (
                <div className="disclosure-section">
                    <button
                        type="button"
                        className={`disclosure-toggle ${accordionOpen ? 'open' : ''}`}
                        onClick={() => setAccordionOpen((prev) => !prev)}
                        aria-expanded={accordionOpen}
                    >
                        <div className="disclosure-label">
                            <span className="disclosure-title">
                                Category Candidates ({otherMatches.length})
                            </span>
                            <span className="disclosure-subtitle">
                                Related classes in {result.category_name || 'category'}
                            </span>
                        </div>
                        <div className="chevron-pill">
                            <svg 
                                className={`chevron-svg ${accordionOpen ? 'rotated' : ''}`}
                                width="14" 
                                height="14" 
                                viewBox="0 0 24 24" 
                                fill="none" 
                                stroke="currentColor" 
                                strokeWidth="2.5" 
                                strokeLinecap="round" 
                                strokeLinejoin="round"
                            >
                                <polyline points="6 9 12 15 18 9" />
                            </svg>
                        </div>
                    </button>

                    <div className={`disclosure-drawer ${accordionOpen ? 'open' : ''}`}>
                        <div className="disclosure-inner-list">
                            {otherMatches.map((m, i) => (
                                <MatchCard key={`${m.label}-${i}`} match={m} rank={i + 2} />
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </section>
    );
}

ResultPanel.propTypes = {
    result: PropTypes.shape({
        class: PropTypes.string,
        confidence: PropTypes.number,
        category_name: PropTypes.string,
        image_path: PropTypes.string,
        matches: PropTypes.arrayOf(
            PropTypes.shape({
                class: PropTypes.string,
                score: PropTypes.number,
                category_name: PropTypes.string,
                image_path: PropTypes.string,
            })
        ),
    }),
    isClassifying: PropTypes.bool,
};
