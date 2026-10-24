export default function FeatureImportanceChart({ data, taskType }) {
  if (!data || !data.length) {
    return (
      <div className="fi-chart">
        <div style={{ textAlign: 'center', color: 'var(--text-subtle)', padding: '20px 0', fontSize: 13 }}>
          Feature attribution data not available
        </div>
      </div>
    );
  }

  const sorted = [...data]
    .sort((a, b) => Math.abs(b.importance ?? b.value ?? 0) - Math.abs(a.importance ?? a.value ?? 0))
    .slice(0, 15);

  const maxVal = Math.max(...sorted.map((d) => Math.abs(d.importance ?? d.value ?? 0)));

  const isShap = taskType === 'regression';

  return (
    <div className="fi-chart">
      {sorted.map((d, i) => {
        const rawVal = Number(d.importance ?? d.value ?? 0);
        const val = isNaN(rawVal) ? 0 : Math.abs(rawVal);
        const pct = maxVal > 0 ? (val / maxVal) * 100 : 0;
        const name = d.feature || d.name || `Feature ${i + 1}`;

        return (
          <div key={i} className="fi-row">
            <div className="fi-label" title={name}>{name}</div>
            <div className="fi-track">
              <div
                className={`fi-fill${isShap ? ' teal' : ''}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="fi-value">
              {val < 0.001 ? val.toExponential(2) : val.toFixed(4)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
