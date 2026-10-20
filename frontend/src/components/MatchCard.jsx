import PropTypes from 'prop-types';

export default function MatchCard({ match, isTopMatch = false, rank = 1 }) {
    const rawConf = match.confidence != null ? match.confidence : match.score;
    const confidencePct = rawConf == null ? null : Math.round(rawConf * 100);
    const confidenceDisplay = rawConf == null ? '—' : `${(rawConf * 100).toFixed(1)}%`;
    
    // Format label for display (turn underscores to spaces for human readability)
    const displayName = (match.label || match.class || 'Unknown').replace(/_/g, ' ');

    // Color code confidence based on physical instrument tones
    const getConfidenceTone = (pct) => {
        if (pct == null) return 'neutral';
        if (pct >= 85) return 'high';
        if (pct >= 60) return 'medium';
        return 'low';
    };

    const tone = getConfidenceTone(confidencePct);

    if (isTopMatch) {
        return (
            <div className={`match-card top-match tone-${tone}`} role="region" aria-label="Top predicted match">
                <div className="top-match-hero">
                    {/* Unobstructed thumbnail image */}
                    <div className="match-thumbnail-wrap">
                        {match.image_url ? (
                            <img src={match.image_url} alt={displayName} className="match-thumbnail" />
                        ) : (
                            <div className="thumbnail-fallback">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                                    <circle cx="8.5" cy="8.5" r="1.5" />
                                    <polyline points="21 15 16 10 5 21" />
                                </svg>
                            </div>
                        )}
                    </div>

                    <div className="top-match-details">
                        {/* Badges placed beside the image, completely unobstructed */}
                        <div className="match-tag-group">
                            <span className="best-match-badge">
                                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                    <polyline points="20 6 9 17 4 12" />
                                </svg>
                                <span>Top Match</span>
                            </span>
                            <span className="match-category-tag">
                                {match.category || match.category_name || 'General'}
                            </span>
                        </div>
                        <h3 className="match-title">{displayName}</h3>

                        <div className="confidence-metric-row">
                            <div className="confidence-numeric">
                                <span className="metric-label">Confidence</span>
                                <span className={`metric-value tone-${tone}`}>{confidenceDisplay}</span>
                            </div>

                            <div className="apple-progress-track">
                                <div
                                    className={`apple-progress-fill tone-${tone}`}
                                    style={{ width: `${Math.min(confidencePct || 0, 100)}%` }}
                                    role="progressbar"
                                    aria-valuenow={confidencePct || 0}
                                    aria-valuemin="0"
                                    aria-valuemax="100"
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="match-card secondary-match">
            <div className="rank-indicator">#{rank}</div>
            <div className="match-thumbnail-mini">
                {match.image_url ? (
                    <img src={match.image_url} alt={displayName} />
                ) : (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <polyline points="21 15 16 10 5 21" />
                    </svg>
                )}
            </div>

            <div className="secondary-match-info">
                <div className="secondary-title-row">
                    <span className="secondary-title">{displayName}</span>
                    <span className={`secondary-score tone-${tone}`}>{confidenceDisplay}</span>
                </div>
                <div className="secondary-meta-row">
                    <span className="secondary-category">{match.category || match.category_name || '—'}</span>
                    <div className="mini-progress-track">
                        <div
                            className={`mini-progress-fill tone-${tone}`}
                            style={{ width: `${Math.min(confidencePct || 0, 100)}%` }}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}

MatchCard.propTypes = {
    match: PropTypes.shape({
        confidence: PropTypes.number,
        score: PropTypes.number,
        image_url: PropTypes.string,
        label: PropTypes.string,
        class: PropTypes.string,
        category: PropTypes.string,
        category_name: PropTypes.string,
    }).isRequired,
    isTopMatch: PropTypes.bool,
    rank: PropTypes.number,
};
