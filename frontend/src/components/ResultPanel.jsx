import { useState } from 'react';
import PropTypes from 'prop-types';
import MatchCard from './MatchCard';

export default function ResultPanel({ result }) {
    const [accordionOpen, setAccordionOpen] = useState(false);

    if (!result) {
        return (
            <section className="glass panel">
                <div className="panel-title">Result</div>
                <div className="result-placeholder">
                    <div className="icon">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="11" cy="11" r="8" />
                            <line x1="21" y1="21" x2="16.65" y2="16.65" />
                        </svg>
                    </div>
                    <span>Classify an image to see results</span>
                </div>
            </section>
        );
    }

    // Construct absolute URL for images
    const getImageUrl = (path) => {
        if (!path) return null;
        return `http://localhost:8000${path}`;
    };

    const topMatch = {
        label: result.class,
        confidence: result.confidence,
        category: result.category_name,
        image_url: getImageUrl(result.image_path)
    };

    const otherMatches = (result.matches || []).map(m => ({
        label: m.class,
        confidence: m.score,
        category: m.category_name,
        image_url: getImageUrl(m.image_path)
    }));

    return (
        <section className="glass panel">
            <div className="panel-title">Result</div>

            <MatchCard match={topMatch} isTopMatch />

            {otherMatches.length > 0 && (
                <div>
                    <button
                        className={`accordion-header ${accordionOpen ? 'open' : ''}`}
                        onClick={() => setAccordionOpen((o) => !o)}
                    >
                        <span>Other matches of the same category ({otherMatches.length})</span>
                        <span className="chevron">▾</span>
                    </button>

                    <div className={`accordion-body ${accordionOpen ? 'open' : ''}`}>
                        {otherMatches.map((m, i) => (
                            <MatchCard key={`${m.label}-${i}`} match={m} />
                        ))}
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
        matches: PropTypes.arrayOf(PropTypes.shape({
            class: PropTypes.string,
            score: PropTypes.number,
            category_name: PropTypes.string,
            image_path: PropTypes.string
        }))
    })
};
