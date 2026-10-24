export default function ConfusionMatrix({ matrix }) {
  if (!matrix || !matrix.length) {
    return (
      <div className="confusion-matrix">
        <div style={{ textAlign: 'center', color: 'var(--text-subtle)', padding: '24px 0', fontSize: 13 }}>
          Confusion matrix available for classification tasks
        </div>
      </div>
    );
  }

  const maxVal = Math.max(...matrix.flat());

  const getCellClass = (row, col, val) => {
    if (row === col) return 'heatmap-diagonal';
    const ratio = maxVal > 0 ? val / maxVal : 0;
    if (ratio > 0.5)  return 'heatmap-high';
    if (ratio > 0.15) return 'heatmap-medium';
    return 'heatmap-low';
  };

  const n = matrix.length;

  return (
    <div className="confusion-matrix">
      <div style={{ marginBottom: 10, fontSize: 11, color: 'var(--text-subtle)', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
        Actual → Predicted
      </div>
      <table className="confusion-table">
        <thead>
          <tr>
            <th />
            {matrix[0].map((_, i) => (
              <th key={i}>P{i}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, ri) => (
            <tr key={ri}>
              <th style={{ textAlign: 'right', paddingRight: 8 }}>A{ri}</th>
              {row.map((val, ci) => (
                <td key={ci} className={getCellClass(ri, ci, val)} title={`Actual ${ri} → Predicted ${ci}: ${val}`}>
                  {val}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 10, display: 'flex', gap: 16, fontSize: 11, color: 'var(--text-subtle)' }}>
        <span><span style={{ display:'inline-block', width:10, height:10, background:'#1d4ed8', borderRadius:2, marginRight:4 }}/>Correct</span>
        <span><span style={{ display:'inline-block', width:10, height:10, background:'#bfdbfe', borderRadius:2, marginRight:4 }}/>Partial</span>
        <span><span style={{ display:'inline-block', width:10, height:10, background:'var(--bg-subtle)', border:'1px solid var(--border-color)', borderRadius:2, marginRight:4 }}/>Incorrect</span>
      </div>
    </div>
  );
}
