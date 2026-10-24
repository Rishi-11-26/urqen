import React, { useState } from 'react';
import { BarChart3, Award } from 'lucide-react';

const CLASSIFICATION_METRICS = [
  { key: 'f1', label: 'F1 Score (Macro)' },
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
  { key: 'roc_auc', label: 'ROC-AUC' },
];

const REGRESSION_METRICS = [
  { key: 'r2', label: 'R² Score' },
  { key: 'mae', label: 'MAE' },
  { key: 'rmse', label: 'RMSE' },
];

export default function ModelComparisonChart({ models = [], primaryMetric = 'f1', taskType = 'binary_classification', bestModel }) {
  const isRegression = taskType.includes('regression');
  const availableMetrics = isRegression ? REGRESSION_METRICS : CLASSIFICATION_METRICS;
  const [selectedMetric, setSelectedMetric] = useState(primaryMetric || (isRegression ? 'r2' : 'f1'));

  const validModels = models.filter((m) => m.status !== 'failed');
  if (validModels.length === 0) return null;

  const isLowerBetter = ['mae', 'mse', 'rmse'].includes(selectedMetric);

  // Compute values
  const data = validModels.map((m) => {
    const holdout = m[selectedMetric] != null ? Number(m[selectedMetric]) : null;
    const cv = selectedMetric === primaryMetric && m.cv_mean != null ? Number(m.cv_mean) : null;
    return {
      model: m.model,
      family: m.family,
      holdout,
      cv,
      cv_std: m.cv_std,
      isWinner: m.model === bestModel,
    };
  });

  // Calculate chart boundaries
  const allValues = data
    .flatMap((d) => [d.holdout, d.cv])
    .filter((v) => v !== null && !isNaN(v) && isFinite(v));

  if (allValues.length === 0) return null;

  let minVal = Math.min(...allValues);
  let maxVal = Math.max(...allValues);

  // For 0-1 metrics (F1, Accuracy, etc.)
  if (!isRegression) {
    minVal = Math.min(0, minVal);
    maxVal = Math.max(1, maxVal);
  } else if (selectedMetric === 'r2') {
    minVal = Math.min(0, minVal);
    maxVal = Math.max(1, maxVal);
  } else {
    minVal = 0;
    maxVal = maxVal * 1.1 || 1;
  }

  const range = maxVal - minVal || 1;

  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-header" style={{ flexWrap: 'wrap', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <BarChart3 size={18} color="var(--primary)" />
          <div>
            <div className="card-title">Model Performance Comparison</div>
            <div className="card-subtitle">
              Visual benchmark of all evaluated algorithms on this dataset
            </div>
          </div>
        </div>

        {/* Metric Selector Tabs */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {availableMetrics.map((met) => (
            <button
              key={met.key}
              onClick={() => setSelectedMetric(met.key)}
              className={`btn btn-sm ${selectedMetric === met.key ? 'btn-primary' : 'btn-ghost'}`}
              style={{ fontSize: 11.5, padding: '4px 10px' }}
            >
              {met.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: '20px 24px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {data.map((d, i) => {
            const val = d.cv !== null ? d.cv : d.holdout;
            const pct = val !== null ? Math.max(0, Math.min(100, ((val - minVal) / range) * 100)) : 0;
            const holdoutPct = d.holdout !== null ? Math.max(0, Math.min(100, ((d.holdout - minVal) / range) * 100)) : 0;

            return (
              <div key={i} style={{ display: 'grid', gridTemplateColumns: '180px 1fr 100px', alignItems: 'center', gap: 16 }}>
                {/* Model Name */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
                  {d.isWinner && <Award size={14} color="#d97706" style={{ flexShrink: 0 }} />}
                  <span
                    style={{
                      fontSize: 12.5,
                      fontWeight: d.isWinner ? 700 : 500,
                      color: d.isWinner ? 'var(--primary)' : 'var(--text-main)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    title={d.model}
                  >
                    {d.model}
                  </span>
                </div>

                {/* Bar Track */}
                <div style={{ position: 'relative', height: 26, background: 'rgba(255, 255, 255, 0.04)', borderRadius: 6, overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
                  {/* CV Score Bar */}
                  <div
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: `${pct}%`,
                      background: d.isWinner
                        ? 'linear-gradient(90deg, #00e5c3 0%, #38bdf8 100%)'
                        : 'linear-gradient(90deg, #1e293b 0%, #334155 100%)',
                      borderRadius: 5,
                      boxShadow: d.isWinner ? '0 0 14px rgba(0, 229, 195, 0.3)' : 'none',
                      transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                    }}
                  />

                  {/* Holdout marker indicator if CV was present */}
                  {d.cv !== null && d.holdout !== null && (
                    <div
                      title={`Holdout Score: ${d.holdout.toFixed(4)}`}
                      style={{
                        position: 'absolute',
                        left: `calc(${holdoutPct}% - 2px)`,
                        top: 2,
                        bottom: 2,
                        width: 4,
                        background: '#fbbf24',
                        borderRadius: 2,
                        boxShadow: '0 0 6px rgba(251, 191, 36, 0.6)',
                        zIndex: 2,
                      }}
                    />
                  )}
                </div>

                {/* Score Number Display */}
                <div style={{ textAlign: 'right', fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 600, color: d.isWinner ? '#00e5c3' : '#94a3b8' }}>
                  {val !== null ? val.toFixed(4) : '—'}
                  {d.cv !== null && (
                    <span style={{ fontSize: 10, color: '#64748b', display: 'block', fontFamily: 'var(--font-mono)' }}>
                      (CV MEAN)
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 20, marginTop: 22, paddingTop: 16, borderTop: '1px solid var(--border-color)', fontSize: 11.5, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 12, height: 12, borderRadius: 3, background: '#00e5c3', boxShadow: '0 0 8px rgba(0, 229, 195, 0.4)' }} />
            <span style={{ color: '#f8fafc' }}>Champion Algorithm</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 12, height: 12, borderRadius: 3, background: '#334155' }} />
            <span>Alternative Baselines</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 4, height: 12, borderRadius: 2, background: '#fbbf24', boxShadow: '0 0 6px #fbbf24' }} />
            <span>Holdout Validation Marker</span>
          </div>
        </div>
      </div>
    </div>
  );
}
