export default function HealthScoreBadge({ score }) {
  if (score === null || score === undefined) return null;

  const pct = Math.round(score * 100);

  let cls, label;
  if (pct >= 85)      { cls = 'excellent'; label = 'Excellent'; }
  else if (pct >= 65) { cls = 'good';      label = 'Good'; }
  else if (pct >= 40) { cls = 'fair';      label = 'Fair'; }
  else                { cls = 'poor';      label = 'Needs Work'; }

  const bg = {
    excellent: '#059669', good: '#0d9488', fair: '#d97706', poor: '#dc2626',
  }[cls];

  return (
    <span className={`health-badge ${cls}`}>
      <span
        className="health-score-circle"
        style={{ background: bg, color: '#fff' }}
      >
        {pct}
      </span>
      {label} Data Quality
    </span>
  );
}
