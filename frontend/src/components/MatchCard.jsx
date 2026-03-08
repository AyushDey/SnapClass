import PropTypes from 'prop-types';

export default function MatchCard({ match, isTopMatch = false }) {
    const confidence = match.confidence == null
        ? '—'
        : (match.confidence * 100).toFixed(1);

    return (
        <div className={`match-card ${isTopMatch ? 'top-match' : ''}`}>
            <div className="match-image">
                {match.image_url ? (
                    <img src={match.image_url} alt={match.label || 'match'} />
                ) : (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <polyline points="21 15 16 10 5 21" />
                    </svg>
                )}
            </div>

            <div className="match-info">
                <div className="item-name">{match.label || 'Unknown'}</div>

                <div className="detail">
                    <span className="label">Confidence</span>
                    <span>{confidence}%</span>
                    <div className="confidence-bar">
                        <div
                            className="fill"
                            style={{ width: `${Math.min(Number.parseFloat(confidence) || 0, 100)}%` }}
                        />
                    </div>
                </div>

                <div className="detail">
                    <span className="label">Category</span>
                    <span>{match.category || '—'}</span>
                </div>
            </div>

            {isTopMatch && <span className="badge">Best Match</span>}
        </div>
    );
}

MatchCard.propTypes = {
    match: PropTypes.shape({
        confidence: PropTypes.number,
        image_url: PropTypes.string,
        label: PropTypes.string,
        category: PropTypes.string
    }).isRequired,
    isTopMatch: PropTypes.bool
};
