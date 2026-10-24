import { useState, useEffect, useRef } from 'react';
import {
  Upload, Database, CheckCircle, AlertTriangle,
  ArrowRight, Trash2, Target, RefreshCw, ShieldCheck,
} from 'lucide-react';
import HealthScoreBadge from '../components/HealthScoreBadge';
import QualityFindingCard from '../components/QualityFindingCard';
import { api } from '../api';

export default function DatasetLabView({
  datasets,
  activeDataset,
  onSelectDataset,
  onDatasetDeleted,
  onDatasetUploaded,
  onProceedToStudio,
}) {
  const [uploading, setUploading]       = useState(false);
  const [uploadError, setUploadError]   = useState(null);
  const [dragOver, setDragOver]         = useState(false);
  const [profile, setProfile]           = useState(null);
  const [audit, setAudit]               = useState(null);
  const [loadingAudit, setLoadingAudit] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [selectedTaskType, setSelectedTaskType] = useState(null);
  const fileInputRef = useRef();

  useEffect(() => {
    if (activeDataset) {
      loadProfileAndAudit(activeDataset.dataset_id, null);
    } else {
      setProfile(null); setAudit(null); setSelectedTarget(null); setSelectedTaskType(null);
    }
  }, [activeDataset]);

  const loadProfileAndAudit = async (datasetId, targetCol) => {
    setLoadingAudit(true);
    try {
      const prof = await api.getDatasetProfile(datasetId, 15);
      setProfile(prof);
      const aud = await api.getDatasetAudit(datasetId, targetCol);
      setAudit(aud);
      setSelectedTarget(targetCol || (aud.target_recommendation?.suggested_target ?? null));
      setSelectedTaskType(aud.target_recommendation?.detected_task ?? 'binary_classification');
    } catch (err) {
      console.error('Failed to load profile/audit:', err);
    } finally {
      setLoadingAudit(false);
    }
  };

  const handleTargetChange = (e) => {
    const t = e.target.value;
    setSelectedTarget(t);
    if (activeDataset) loadProfileAndAudit(activeDataset.dataset_id, t);
  };

  const uploadFile = async (file) => {
    if (!file) return;
    setUploading(true); setUploadError(null);
    try {
      const res = await api.uploadDataset(file);
      onDatasetUploaded(res);
      onSelectDataset(res);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleFileInput  = (e) => uploadFile(e.target.files?.[0]);
  const handleDrop       = (e) => { e.preventDefault(); setDragOver(false); uploadFile(e.dataTransfer.files?.[0]); };
  const handleDragOver   = (e) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave  = () => setDragOver(false);

  const handleDelete = async (datasetId) => {
    if (!window.confirm('Delete this dataset? This cannot be undone.')) return;
    try {
      await api.deleteDataset(datasetId);
      onDatasetDeleted(datasetId);
    } catch (err) {
      alert(`Deletion failed: ${err.message}`);
    }
  };

  const totalIssues = (audit?.quality_findings ?? []).filter((f) => f.severity !== 'pass').length;

  return (
    <div className="view-container anim-fade">
      <div className="dataset-intro">
        <div>
          <div className="eyebrow">DATASET LAB / DATA INSPECTION</div>
          <h1 className="page-title">Dataset Lab</h1>
          <p className="page-subtitle">Upload a table, inspect its structure, and review quality signals before configuring an experiment.</p>
        </div>
        {activeDataset && (
          <div className="dataset-active-label">
            <Database size={15} />
            <span title={activeDataset.name || activeDataset.filename}>
              {activeDataset.name || activeDataset.filename}
            </span>
          </div>
        )}
      </div>
      {/* Top strip: upload + dataset list */}
      <div className="dataset-top-grid">

        {/* ── Dropzone ──────────────────────────────────── */}
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-header">
            <div>
              <div className="card-title">Upload Dataset</div>
              <div className="card-subtitle">CSV · TSV · XLSX · XLS · up to 50 MB</div>
            </div>
          </div>

          <div style={{ padding: '16px 20px 20px' }}>
            <label
              className={`dropzone${dragOver ? ' active' : ''}`}
              style={{ display: 'block', cursor: uploading ? 'wait' : 'pointer' }}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.tsv,.xlsx,.xls"
                onChange={handleFileInput}
                style={{ display: 'none' }}
                disabled={uploading}
              />
              <div className="dropzone-icon">
                <Upload size={22} color="var(--primary)" />
              </div>
              {uploading ? (
                <>
                  <div className="dropzone-title" style={{ color: 'var(--primary)' }}>Uploading &amp; Profiling…</div>
                  <div className="dropzone-sub">This may take a moment for large files</div>
                  <div className="spinner" style={{ margin: '12px auto 0' }} />
                </>
              ) : (
                <>
                  <div className="dropzone-title">Drag file here or click to browse</div>
                  <div className="dropzone-sub">Schema detected automatically on upload</div>
                </>
              )}
            </label>

            {uploadError && (
              <div className="alert alert-danger" style={{ marginTop: 12, marginBottom: 0 }}>
                <AlertTriangle size={15} className="alert-icon" />
                {uploadError}
              </div>
            )}
          </div>
        </div>

        {/* ── Dataset Switcher ──────────────────────────── */}
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-header">
            <div>
              <div className="card-title">Workspace Datasets</div>
              <div className="card-subtitle">{datasets.length} dataset{datasets.length !== 1 ? 's' : ''} available — click to inspect</div>
            </div>
            {activeDataset && (
              <button className="btn btn-ghost btn-sm" title="Refresh" onClick={() => loadProfileAndAudit(activeDataset.dataset_id, selectedTarget)}>
                <RefreshCw size={14} />
              </button>
            )}
          </div>

          {datasets.length === 0 ? (
            <div className="empty-state" style={{ padding: '28px 20px' }}>
              <div className="empty-icon"><Database size={22} /></div>
              <div className="empty-title">No datasets yet</div>
              <div className="empty-desc">Upload a CSV file to begin data quality inspection.</div>
            </div>
          ) : (
            <div style={{ padding: '10px 16px 14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: 10, maxHeight: 190, overflowY: 'auto' }}>
              {datasets.map((ds) => {
                const isSelected = activeDataset?.dataset_id === ds.dataset_id;
                return (
                  <div
                    key={ds.dataset_id}
                    role="button"
                    tabIndex={0}
                    aria-pressed={isSelected}
                    onClick={() => onSelectDataset(ds)}
                    onKeyDown={(event) => {
                      if (event.target !== event.currentTarget) return;
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        onSelectDataset(ds);
                      }
                    }}
                    className={`dataset-picker${isSelected ? ' is-selected' : ''}`}
                    style={{
                      padding: '10px 12px',
                      borderRadius: 'var(--radius-sm)',
                      border: `1.5px solid ${isSelected ? 'var(--primary)' : 'var(--border-color)'}`,
                      background: isSelected ? 'var(--primary-light)' : 'var(--bg-subtle)',
                      transition: 'all var(--duration-fast)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 6 }}>
                      <strong style={{ fontSize: 13, color: isSelected ? 'var(--primary)' : 'var(--text-main)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }} title={ds.name || ds.filename}>
                        {ds.name || ds.filename || 'Dataset'}
                      </strong>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(ds.dataset_id); }}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-subtle)', padding: 2, flexShrink: 0 }}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 5 }}>
                      {(ds.row_count ?? ds.n_rows ?? 0).toLocaleString()} rows · {ds.column_count ?? ds.n_columns ?? 0} cols
                    </div>
                    <span className="badge badge-neutral" style={{ marginTop: 6, fontSize: 10 }}>
                      {(ds.file_type || ds.file_format || 'csv').toUpperCase()}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Active Dataset Detail */}
      {activeDataset && (
        <>
          {/* Target + Health Row */}
          <div className="grid-2" style={{ marginBottom: 20 }}>
            {/* Target */}
            <div className="card" style={{ marginBottom: 0 }}>
              <div className="card-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Target size={16} color="var(--primary)" />
                  <div>
                    <div className="card-title">Target &amp; Task Detection</div>
                    <div className="card-subtitle">URQEN can suggest a prediction target from your data</div>
                  </div>
                </div>
                {loadingAudit && <div className="spinner" />}
              </div>

              <div className="card-body">
                {audit?.target_recommendation ? (
                  <div>
                    <div style={{ display: 'flex', gap: 12, marginBottom: 14, alignItems: 'flex-end', flexWrap: 'wrap' }}>
                      <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: 200 }}>
                        <label className="form-label">Target Column (Prediction Goal)</label>
                        <select className="form-select" value={selectedTarget || ''} onChange={handleTargetChange}>
                          {audit.target_recommendation.candidate_targets.map((col) => (
                            <option key={col} value={col}>
                              {col}{col === audit.target_recommendation.suggested_target ? ' ★ Suggested' : ''}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="form-group" style={{ marginBottom: 0, width: 230 }}>
                        <label className="form-label">Task Type (Auto-Detected)</label>
                        <select
                          className="form-select"
                          value={selectedTaskType || audit.target_recommendation.detected_task}
                          onChange={(e) => setSelectedTaskType(e.target.value)}
                        >
                          <option value="regression">Regression (Continuous Values)</option>
                          <option value="binary_classification">Binary Classification (2 Classes)</option>
                          <option value="multiclass_classification">Multiclass Classification (3+ Classes)</option>
                        </select>
                      </div>
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5, background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', padding: '8px 12px' }}>
                      <strong>Why?</strong> {audit.target_recommendation.reasons?.join('; ')}
                    </div>
                  </div>
                ) : (
                  <div style={{ color: 'var(--text-subtle)', fontSize: 13, textAlign: 'center', padding: '12px 0' }}>
                    {loadingAudit ? 'Analyzing columns…' : 'No target detected'}
                  </div>
                )}
              </div>
            </div>

            {/* Health Score Card */}
            <div className="card" style={{ marginBottom: 0 }}>
              <div className="card-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <ShieldCheck size={16} color="var(--success)" />
                  <div>
                    <div className="card-title">Data Health Score</div>
                    <div className="card-subtitle">8-point inspection result</div>
                  </div>
                </div>
                {audit && <HealthScoreBadge score={audit.health_score} />}
              </div>

              <div className="card-body">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--primary)', letterSpacing: '-1px' }}>
                      {audit ? Math.round((audit.health_score ?? 0) * 100) : '—'}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>Health Score</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 24, fontWeight: 800, color: totalIssues > 0 ? 'var(--warning)' : 'var(--success)', letterSpacing: '-1px' }}>
                      {totalIssues}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>Issues Found</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-1px' }}>
                      {profile?.columns?.length ?? '—'}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>Columns</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Proceed CTA */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            background: 'linear-gradient(135deg, rgba(0, 229, 195, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%)',
            border: '1.5px solid rgba(0, 229, 195, 0.3)',
            borderRadius: 'var(--radius-md)',
            padding: '18px 24px',
            marginBottom: 20,
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
          }}>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700, color: '#f8fafc', marginBottom: 3 }}>
                Ready to benchmark machine learning models?
              </div>
              <div style={{ fontSize: 12.5, color: '#94a3b8' }}>
                Target column selected. Launch Guided Mode for automated benchmarking or Expert Mode for hyperparameter control.
              </div>
            </div>
            <button
              className="btn btn-primary"
              style={{ flexShrink: 0, padding: '10px 20px' }}
              onClick={() => onProceedToStudio(activeDataset, selectedTarget, selectedTaskType)}
            >
              Proceed to Experiment Studio
              <ArrowRight size={15} />
            </button>
          </div>

          {/* Quality Findings */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Data quality findings</div>
                <div className="card-subtitle">Heuristic and statistical checks on the selected dataset</div>
              </div>
              <span className={`badge ${!audit ? 'badge-neutral' : totalIssues === 0 ? 'badge-success' : 'badge-warning'}`}>
                {!audit ? 'Audit unavailable' : totalIssues === 0 ? 'No findings returned' : `${totalIssues} issue${totalIssues !== 1 ? 's' : ''} detected`}
              </span>
            </div>

            <div style={{ padding: '14px 18px' }}>
              {audit?.quality_findings?.length > 0 ? (
                audit.quality_findings.map((f, i) => <QualityFindingCard key={i} finding={f} />)
              ) : loadingAudit ? (
                <div style={{ padding: '20px 4px', color: 'var(--text-muted)', fontSize: 13 }}>Checking dataset quality…</div>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '20px 4px', color: 'var(--success)', fontSize: 13.5, fontWeight: 600 }}>
                  <CheckCircle size={20} />
                  {audit ? 'The audit returned no quality findings.' : 'Quality findings are not available yet.'}
                </div>
              )}
            </div>
          </div>

          {/* Preview Table */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Dataset Preview</div>
                <div className="card-subtitle">First 15 rows · {profile?.columns?.length ?? 0} columns</div>
              </div>
            </div>

            {profile?.preview_rows ? (
              <div className="table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      {profile.columns.map((col) => (
                        <th key={col.name}>
                          {col.name}
                          <div style={{ fontSize: 10, color: 'var(--text-subtle)', fontWeight: 500, textTransform: 'lowercase', letterSpacing: 0, marginTop: 1 }}>
                            {col.detected_type}
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {profile.preview_rows.map((row, ri) => (
                      <tr key={ri}>
                        {profile.columns.map((col) => (
                          <td key={col.name} className="text-mono" style={{ fontSize: 12.5 }}>
                            {row[col.name] !== null && row[col.name] !== undefined ? String(row[col.name]) : <span style={{ color: 'var(--text-subtle)' }}>null</span>}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ padding: 28, textAlign: 'center', color: 'var(--text-subtle)', fontSize: 13 }}>
                {loadingAudit ? 'Loading dataset preview…' : 'Preview unavailable'}
              </div>
            )}
          </div>
        </>
      )}

      {/* No dataset selected yet */}
      {!activeDataset && datasets.length > 0 && (
        <div className="card">
          <div className="empty-state">
            <div className="empty-icon"><Database size={24} /></div>
            <div className="empty-title">Select a dataset to begin inspection</div>
            <div className="empty-desc">Click any dataset in the workspace panel above to load its 8-point quality audit.</div>
          </div>
        </div>
      )}
    </div>
  );
}
