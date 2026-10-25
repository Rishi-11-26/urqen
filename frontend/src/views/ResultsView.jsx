import {
  Award, FileText, Package, Download, ClipboardCheck,
  BarChart3, TrendingUp,
} from 'lucide-react';
import ConfusionMatrix from '../components/ConfusionMatrix';
import FeatureImportanceChart from '../components/FeatureImportanceChart';
import ModelComparisonChart from '../components/ModelComparisonChart';
import { api } from '../api';

const fmtNum = (val, digits = 4, fallback = '—') => {
  if (val == null || val === '') return fallback;
  const n = Number(val);
  return isNaN(n) ? String(val) : n.toFixed(digits);
};

export default function ResultsView({ experiment, onSelectAnotherExperiment }) {
  if (!experiment) {
    return (
      <div className="view-container">
        <div className="empty-state">
          <div className="empty-icon"><BarChart3 size={26} /></div>
          <div className="empty-title">No Active Experiment</div>
          <div className="empty-desc">Run a benchmark in Experiment Studio or select a past run from Run History.</div>
          <button className="btn btn-primary" onClick={onSelectAnotherExperiment}>
            Browse Run History
          </button>
        </div>
      </div>
    );
  }

  const {
    experiment_id, dataset_name, task_type, target_column,
    best_model, winner_explanation,
  } = experiment;

  const resObj         = experiment?.results || {};
  const modelsList     = resObj.models || experiment?.models || [];
  const primaryMetric  = resObj.primary_metric || (task_type?.includes('regression') ? 'r2' : 'f1') || 'score';
  const featImportance = resObj.feature_importance;
  const confusionData  = resObj.confusion_matrix;

  const winnerModel = modelsList.find((m) => m.model === best_model) || modelsList[0];
  const winnerScore = winnerModel?.cv_mean ?? winnerModel?.[primaryMetric] ?? winnerModel?.f1 ?? winnerModel?.r2 ?? winnerModel?.accuracy;

  const metricLabel = primaryMetric.toUpperCase();

  return (
    <div className="view-container anim-fade">
      <div className="results-intro">
        <div>
          <div className="eyebrow">RESULTS / EXPERIMENT REVIEW</div>
          <h1 className="page-title">Model results</h1>
          <p className="page-subtitle">
            Compare the metrics and diagnostics recorded for this experiment.
          </p>
        </div>
        <span className="badge badge-neutral">{modelsList.length} models evaluated</span>
      </div>
      {/* ── Winner Banner ──────────────────────────────────── */}
      <div className="winner-banner" style={{ marginBottom: 20 }}>
        <div className="winner-trophy"><Award size={23} /></div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'rgba(255,255,255,0.6)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 5 }}>
            Best model in this run
          </div>
          <div className="winner-model-name">{best_model || 'Benchmark Complete'}</div>
          <div style={{ display: 'flex', gap: 10, marginTop: 8, flexWrap: 'wrap' }}>
            {winnerScore !== undefined && (
              <div className="winner-score-chip">
                {winnerModel?.cv_mean != null ? `CV ${metricLabel}` : `${metricLabel} (Holdout)`}: {fmtNum(winnerScore, 4)}
              </div>
            )}
            <div className="winner-score-chip">
              {dataset_name} → {target_column}
            </div>
            <div className="winner-score-chip" style={{ textTransform: 'capitalize' }}>
              {task_type?.replace(/_/g, ' ')}
            </div>
          </div>
        </div>

        {/* Downloads */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, flexShrink: 0 }}>
          <a
            href={api.downloadReportUrl(experiment_id)}
            target="_blank" rel="noreferrer"
            className="btn"
            style={{ background: 'rgba(255,255,255,0.06)', color: '#fff', border: '1px solid rgba(255,255,255,0.15)', fontSize: 12.5 }}
          >
            <FileText size={14} color="#00e5c3" /> HTML Report
          </a>
          {resObj.evaluation_available && (
            <a
            href={api.downloadEvaluationUrl(experiment_id)}
            target="_blank" rel="noreferrer"
            className="btn"
            style={{ background: 'rgba(255,255,255,0.06)', color: '#fff', border: '1px solid rgba(255,255,255,0.15)', fontSize: 12.5 }}
            title="Download the stored holdout predictions, metrics, split audit, and verification results"
          >
            <ClipboardCheck size={14} color="#fbbf24" /> Evaluation JSON
            </a>
          )}
          <a
            href={api.downloadBundleUrl(experiment_id)}
            target="_blank" rel="noreferrer"
            className="btn"
            style={{ background: 'rgba(255,255,255,0.06)', color: '#fff', border: '1px solid rgba(255,255,255,0.15)', fontSize: 12.5 }}
          >
            <Package size={14} color="#38bdf8" /> Model Bundle
          </a>
          <a
            href={api.downloadPreprocessedDataUrl(experiment_id)}
            target="_blank" rel="noreferrer"
            className="btn"
            style={{ background: 'rgba(255,255,255,0.06)', color: '#fff', border: '1px solid rgba(255,255,255,0.15)', fontSize: 12.5 }}
            title="Download clean preprocessed CSV with original schema format"
          >
            <Download size={14} color="#a855f7" /> Preprocessed CSV
          </a>
        </div>
      </div>

      {/* Winner explanation */}
      {winner_explanation && (
        <div className="alert alert-info" style={{ marginBottom: 20 }}>
          <TrendingUp size={15} className="alert-icon" />
          <span>{winner_explanation}</span>
        </div>
      )}

      {/* ── Auto-Generated Model Comparison Chart ──────────── */}
      <ModelComparisonChart
        models={modelsList}
        primaryMetric={primaryMetric}
        taskType={task_type}
        bestModel={best_model}
      />

      {/* ── Leaderboard ────────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Model Leaderboard</div>
            <div className="card-subtitle">
              Holdout evaluation and cross-validation variance — Dataset: <strong>{dataset_name}</strong> · Target: <strong>{target_column}</strong>
            </div>
          </div>
          <span className="badge badge-neutral">{modelsList.length} models evaluated</span>
        </div>

        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Family</th>
                <th>{metricLabel} (Holdout)</th>
                <th>CV Mean ± Std</th>
                <th>Train Time</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {[...modelsList]
                .sort((a, b) => {
                  if (a.model === best_model) return -1;
                  if (b.model === best_model) return 1;
                  const isLowerBetter = ['mae', 'mse', 'rmse'].includes(primaryMetric);
                  const aScore = a.cv_mean != null ? a.cv_mean : (a[primaryMetric] ?? (isLowerBetter ? Infinity : -Infinity));
                  const bScore = b.cv_mean != null ? b.cv_mean : (b[primaryMetric] ?? (isLowerBetter ? Infinity : -Infinity));
                  return isLowerBetter ? aScore - bScore : bScore - aScore;
                })
                .map((m, idx) => {
                  const isWinner = m.model === best_model;
                  return (
                    <tr key={idx} className={isWinner ? 'row-winner' : ''}>
                      <td>
                        <span style={{
                          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                          width: 24, height: 24, borderRadius: '50%', fontSize: 11, fontWeight: 700,
                          fontFamily: 'var(--font-mono)',
                          background: idx === 0 ? 'rgba(0, 229, 195, 0.15)' : idx === 1 ? 'rgba(99, 102, 241, 0.15)' : idx === 2 ? 'rgba(245, 158, 11, 0.15)' : 'rgba(255, 255, 255, 0.04)',
                          color: idx === 0 ? '#00e5c3' : idx === 1 ? '#818cf8' : idx === 2 ? '#fbbf24' : '#64748b',
                          border: idx === 0 ? '1px solid rgba(0, 229, 195, 0.4)' : '1px solid rgba(255, 255, 255, 0.06)',
                        }}>{idx + 1}</span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                          {isWinner && <Award size={14} color="#d97706" />}
                          <span style={{ fontWeight: isWinner ? 700 : 500 }}>{m.model}</span>
                        </div>
                      </td>
                      <td style={{ color: 'var(--text-muted)', fontSize: 12.5 }}>{m.family || '—'}</td>
                      <td className="text-mono" style={{ fontWeight: 700, color: idx === 0 ? 'var(--primary)' : 'var(--text-main)' }}>
                        {m[primaryMetric] !== undefined ? fmtNum(m[primaryMetric], 4) : '—'}
                      </td>
                      <td className="text-mono" style={{ color: 'var(--text-muted)' }}>
                        {m.cv_mean != null ? `${fmtNum(m.cv_mean, 3)} ± ${fmtNum(m.cv_std || 0, 3)}` : '—'}
                      </td>
                      <td className="text-mono" style={{ color: 'var(--text-muted)' }}>
                        {m.training_time_seconds != null ? `${fmtNum(m.training_time_seconds, 2)}s` : '—'}
                      </td>
                      <td>
                        <span className={`badge ${m.status === 'success' ? 'badge-success' : 'badge-danger'}`} style={{ fontSize: 11 }}>
                          {m.status || 'success'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Confusion Matrix + Feature Importance ─────────── */}
      <div className="grid-2">
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-header">
            <div>
              <div className="card-title">Holdout Confusion Matrix</div>
              <div className="card-subtitle">Diagonal = true positives</div>
            </div>
          </div>
          <ConfusionMatrix matrix={confusionData} />
        </div>

        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-header">
            <div>
              <div className="card-title">Feature Importance & Attribution</div>
              <div className="card-subtitle">Global contribution to model predictions</div>
            </div>
          </div>
          <FeatureImportanceChart data={featImportance?.feature_importances} taskType={task_type} />
        </div>
      </div>
    </div>
  );
}
