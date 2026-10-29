import React, { useState, useRef } from 'react';
import { X, PlayCircle, Upload, CheckCircle2, AlertTriangle, Download, Lock } from 'lucide-react';
import { api } from '../api';

export default function TestPredictModal({ isOpen, onClose, experimentId, bestModel, onUpgrade }) {
  const [file, setFile] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const fileInputRef = useRef(null);

  if (!isOpen) return null;

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setError(null);
      setResults(null);
    }
  };

  const handleRunPredict = async () => {
    if (!file) return;
    setRunning(true);
    setError(null);
    try {
      const resp = await api.predictWithCsv(experimentId, file);
      setResults(resp);
    } catch (err) {
      if (err.message && (err.message.includes('403') || err.message.includes('Pro feature'))) {
        onUpgrade?.();
        onClose();
        return;
      }
      setError(err.message || 'Prediction failed.');
    } finally {
      setRunning(false);
    }
  };

  const handleDownloadPredictionsCsv = () => {
    if (!results || !results.records) return;
    const header = ['sample_index', 'prediction'];
    const rows = results.records.map((r, i) => [i + 1, r.prediction]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [header.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `urqen_predictions_${experimentId.slice(0, 8)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: 'rgba(11, 17, 32, 0.8)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
        animation: 'fadeIn 0.2s ease-out',
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: '100%',
          maxWidth: 680,
          maxHeight: '90vh',
          background: 'var(--bg-card)',
          borderRadius: 'var(--radius-lg)',
          boxShadow: 'var(--shadow-xl)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          animation: 'slideUp 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '16px 24px',
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'var(--bg-subtle)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 'var(--radius-sm)',
                background: 'linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%)',
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <PlayCircle size={18} />
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-main)' }}>
                Test Predictions with {bestModel || 'Trained Model'}
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                Upload an unlabelled CSV dataset to generate predictions through the fitted pipeline
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--text-muted)',
              padding: 4,
              borderRadius: 6,
            }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div style={{ padding: '24px', flex: 1, overflowY: 'auto' }}>
          {!results ? (
            <div>
              <div
                style={{
                  border: '2px dashed var(--border-color)',
                  borderRadius: 'var(--radius-md)',
                  padding: '24px 16px',
                  textAlign: 'center',
                  background: 'var(--bg-subtle)',
                  cursor: 'pointer',
                  marginBottom: 16,
                }}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
                <Upload size={24} color="var(--primary)" style={{ margin: '0 auto 8px', display: 'block' }} />
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-main)' }}>
                  {file ? file.name : 'Click to select an unlabelled CSV file'}
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 4 }}>
                  {file ? `${(file.size / 1024).toFixed(1)} KB` : 'Must contain identical feature columns as the training set'}
                </div>
              </div>

              {error && (
                <div className="alert alert-danger" style={{ marginBottom: 16 }}>
                  <AlertTriangle size={15} className="alert-icon" />
                  {error}
                </div>
              )}

              <button
                className="btn btn-primary"
                disabled={!file || running}
                onClick={handleRunPredict}
                style={{ width: '100%', justifyContent: 'center', gap: 8, padding: '11px 0' }}
              >
                {running ? <div className="spinner" style={{ width: 14, height: 14 }} /> : <PlayCircle size={16} />}
                {running ? 'Generating Predictions…' : 'Generate Model Predictions'}
              </button>
            </div>
          ) : (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <CheckCircle2 size={18} color="var(--success)" />
                  <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-main)' }}>
                    Generated {results.predictions_count || results.records?.length || 0} Predictions
                  </span>
                </div>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleDownloadPredictionsCsv}
                  style={{ gap: 6, fontSize: 12 }}
                >
                  <Download size={13} /> Download Predictions CSV
                </button>
              </div>

              {/* Sample Table */}
              <div style={{ maxHeight: 280, overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: 'var(--bg-subtle)', borderBottom: '1px solid var(--border-color)', textAlign: 'left' }}>
                      <th style={{ padding: '8px 12px', fontWeight: 600, color: 'var(--text-muted)' }}>#</th>
                      <th style={{ padding: '8px 12px', fontWeight: 600, color: 'var(--text-main)' }}>Predicted Target</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(results.records || []).slice(0, 15).map((rec, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--border-light)' }}>
                        <td style={{ padding: '6px 12px', color: 'var(--text-muted)' }}>{i + 1}</td>
                        <td style={{ padding: '6px 12px', fontWeight: 600, color: 'var(--primary)' }}>
                          {String(rec.prediction)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <button
                className="btn btn-secondary btn-sm"
                onClick={() => { setResults(null); setFile(null); }}
                style={{ marginTop: 14 }}
              >
                Score Another CSV
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '12px 24px',
            borderTop: '1px solid var(--border-color)',
            background: 'var(--bg-subtle)',
            display: 'flex',
            justifyContent: 'flex-end',
          }}
        >
          <button className="btn btn-secondary btn-sm" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
