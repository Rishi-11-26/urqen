import { useState, useEffect } from 'react';
import {
  Sparkles, Sliders, Play, CheckSquare, Square,
  AlertCircle, Settings2, ChevronDown, ChevronUp, Lock,
  Layers, AlertTriangle,
} from 'lucide-react';
import { api } from '../api';

const CLASSIFICATION_MODELS = [
  { name: 'Logistic Regression',          family: 'Linear',                defaultSelected: true },
  { name: 'Ridge Classifier',             family: 'Linear',                defaultSelected: false },
  { name: 'SGD Classifier',               family: 'Linear',                defaultSelected: false },
  { name: 'Decision Tree',                family: 'Tree & Ensemble',       defaultSelected: true },
  { name: 'Random Forest',                family: 'Tree & Ensemble',       defaultSelected: true },
  { name: 'Extra Trees',                  family: 'Tree & Ensemble',       defaultSelected: false },
  { name: 'Gradient Boosting',            family: 'Tree & Ensemble',       defaultSelected: false },
  { name: 'AdaBoost',                     family: 'Tree & Ensemble',       defaultSelected: false },
  { name: 'K-Nearest Neighbors',          family: 'Distance & Kernel',     defaultSelected: false },
  { name: 'Linear SVM',                   family: 'Distance & Kernel',     defaultSelected: false },
  { name: 'Kernel SVM',                   family: 'Distance & Kernel',     defaultSelected: false },
  { name: 'Naive Bayes',                  family: 'Probabilistic',         defaultSelected: false },
  { name: 'RBF Network (RBFN)',           family: 'Distance & Kernel',     defaultSelected: false },
  { name: 'XGBoost',                      family: 'Gradient Boosting',     defaultSelected: true },
];

const REGRESSION_MODELS = [
  { name: 'Linear Regression',            family: 'Linear',                defaultSelected: true },
  { name: 'Ridge',                        family: 'Linear',                defaultSelected: true },
  { name: 'Lasso',                        family: 'Linear',                defaultSelected: false },
  { name: 'ElasticNet',                   family: 'Linear',                defaultSelected: false },
  { name: 'Decision Tree Regressor',      family: 'Tree & Ensemble',       defaultSelected: false },
  { name: 'Random Forest Regressor',      family: 'Tree & Ensemble',       defaultSelected: true },
  { name: 'Extra Trees Regressor',        family: 'Tree & Ensemble',       defaultSelected: false },
  { name: 'Gradient Boosting Regressor',  family: 'Tree & Ensemble',       defaultSelected: false },
  { name: 'AdaBoost Regressor',           family: 'Tree & Ensemble',       defaultSelected: false },
  { name: 'K-Nearest Neighbors Regressor',family: 'Distance & Kernel',    defaultSelected: false },
  { name: 'Support Vector Regressor',     family: 'Distance & Kernel',     defaultSelected: false },
  { name: 'XGBoost Regressor',            family: 'Gradient Boosting',     defaultSelected: true },
];

const FAMILY_COLORS = {
  'Linear':           '#00e5c3',
  'Tree & Ensemble':  '#38bdf8',
  'Distance & Kernel':'#f59e0b',
  'Probabilistic':    '#a855f7',
  'Gradient Boosting':'#6366f1',
};

function HyperParamInput({ label, hint, value, onChange, type = 'number', disabled = false, ...rest }) {
  return (
    <div className="form-group" style={{ marginBottom: 0 }}>
      <label className="form-label" style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
        {label}
        {hint && <span className="form-label-hint">{hint}</span>}
      </label>
      <input
        type={type}
        className="form-input"
        value={value}
        onChange={onChange}
        disabled={disabled}
        style={{
          fontSize: 13,
          cursor: disabled ? 'not-allowed' : 'auto',
          backgroundColor: disabled ? 'var(--bg-subtle)' : undefined,
          opacity: disabled ? 0.7 : 1,
        }}
        {...rest}
      />
    </div>
  );
}

export default function ExperimentStudioView({
  activeDataset,
  targetColumn,
  initialTaskType,
  mode,
  setMode,
  onExperimentCompleted,
  isBenchmarking = false,
  setIsBenchmarking,
  maintenanceMode = false,
}) {
  const [selectedTarget, setSelectedTarget]   = useState(targetColumn || '');
  const [candidateTargets, setCandidateTargets] = useState([]);
  const [allColumns, setAllColumns]             = useState([]);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [leakageColumns, setLeakageColumns]     = useState([]);
  const [taskType, setTaskType]           = useState(initialTaskType || 'binary_classification');
  const [candidateModels, setCandidateModels] = useState([]);
  const [selectedModels, setSelectedModels]   = useState([]);
  const [showHyperparams, setShowHyperparams] = useState(false);

  const [cvFolds, setCvFolds]       = useState(10);
  const [testSplit, setTestSplit]   = useState(20);
  const [scalerType, setScalerType] = useState('standard');

  const [xgboostParams, setXgboostParams] = useState({ learning_rate: 0.1, max_depth: 6, n_estimators: 100, min_child_weight: 1 });
  const [rfParams,  setRfParams]  = useState({ n_estimators: 100, max_depth: 10 });
  const [dtParams,  setDtParams]  = useState({ max_depth: 8, min_samples_split: 2 });
  const [svmParams, setSvmParams] = useState({ C: 1.0 });

  const [running,             setRunning]             = useState(isBenchmarking);
  const [activeExpId,         setActiveExpId]         = useState(null);
  const [runProgressMessage,  setRunProgressMessage]  = useState('');
  const [errorMessage,        setErrorMessage]        = useState(null);

  useEffect(() => {
    if (activeDataset?.dataset_id) {
      const targetQuery = targetColumn || null;
      api.getDatasetAudit(activeDataset.dataset_id, targetQuery)
        .then((aud) => {
          const rec = aud.target_recommendation;
          const cols = rec?.candidate_targets || [];
          if (cols.length > 0) {
            setCandidateTargets(cols);
            setAllColumns(cols);
          }
          const chosen = targetColumn || rec?.suggested_target || cols[0] || '';
          setSelectedTarget(chosen);
          const detected = rec?.detected_task || 'binary_classification';
          setTaskType(detected);

          // Extract leakage features from audit warnings
          const leakList = [];
          (aud.leakage_warnings || []).forEach((w) => {
            if (w.feature) leakList.push(w.feature);
          });
          setLeakageColumns(leakList);

          // Default input features: all candidate columns except the chosen target
          if (cols.length > 0) {
            setSelectedFeatures(cols.filter((c) => c !== chosen));
          }
        })
        .catch((err) => {
          console.warn('Dataset audit fetch error, trying profile fallback:', err);
          api.getDatasetProfile(activeDataset.dataset_id)
            .then((prof) => {
              if (prof?.columns?.length > 0) {
                const cols = prof.columns.map((c) => c.name);
                setCandidateTargets(cols);
                setAllColumns(cols);
                const chosen = targetColumn || cols[cols.length - 1] || '';
                setSelectedTarget(chosen);
                setSelectedFeatures(cols.filter((c) => c !== chosen));
              }
            })
            .catch(() => {});
        });
    }
  }, [activeDataset, targetColumn]);

  const handleTargetChange = (e) => {
    const newTarget = e.target.value;
    setSelectedTarget(newTarget);
    // Remove new target from selected features if present
    setSelectedFeatures((prev) => prev.filter((f) => f !== newTarget));
    if (activeDataset?.dataset_id && newTarget) {
      api.getDatasetAudit(activeDataset.dataset_id, newTarget)
        .then((aud) => {
          if (aud.target_recommendation?.detected_task) {
            setTaskType(aud.target_recommendation.detected_task);
          }
          const leakList = [];
          (aud.leakage_warnings || []).forEach((w) => {
            if (w.feature) leakList.push(w.feature);
          });
          setLeakageColumns(leakList);
        })
        .catch(() => {});
    }
  };

  const toggleFeature = (colName) => {
    if (running) return;
    setSelectedFeatures((prev) =>
      prev.includes(colName) ? prev.filter((c) => c !== colName) : [...prev, colName]
    );
  };

  useEffect(() => {
    const list = taskType.includes('regression') ? REGRESSION_MODELS : CLASSIFICATION_MODELS;
    setCandidateModels(list);
    setSelectedModels(list.filter((m) => m.defaultSelected).map((m) => m.name));
  }, [taskType]);

  useEffect(() => {
    let iv = null;
    if (running && activeExpId) {
      iv = setInterval(async () => {
        try {
          const exp = await api.getExperiment(activeExpId);
          if (exp.status === 'completed') {
            setRunning(false);
            setIsBenchmarking?.(false);
            clearInterval(iv);
            onExperimentCompleted(exp);
          } else if (exp.status === 'failed') {
            setRunning(false);
            setIsBenchmarking?.(false);
            setErrorMessage(exp.error_message || 'Experiment failed.');
            clearInterval(iv);
          } else {
            setRunProgressMessage(`Benchmarking ${selectedModels.length} estimators — cross-validation in progress…`);
          }
        } catch (e) { console.error(e); }
      }, 1500);
    }
    return () => clearInterval(iv);
  }, [running, activeExpId]);

  const toggleModel = (name) => {
    if (running) return;
    if (selectedModels.includes(name)) {
      if (selectedModels.length > 1) setSelectedModels(selectedModels.filter((m) => m !== name));
    } else {
      setSelectedModels([...selectedModels, name]);
    }
  };

  const handleLaunch = async () => {
    if (running) return;
    if (!activeDataset) { alert('Select a dataset first in Dataset Lab.'); return; }
    const finalTarget = selectedTarget || targetColumn;
    if (!finalTarget) {
      alert('Please select a target column to predict.');
      return;
    }
    if (selectedFeatures.length === 0) {
      alert('Please select at least one input feature column to train the models.');
      return;
    }

    setRunning(true);
    setIsBenchmarking?.(true);
    setErrorMessage(null);
    setRunProgressMessage('Initializing experiment pipeline and holdout partitions…');

    const hp = {};
    if (selectedModels.includes('XGBoost') || selectedModels.includes('XGBoost Regressor')) {
      const k = selectedModels.includes('XGBoost') ? 'XGBoost' : 'XGBoost Regressor';
      hp[k] = { learning_rate: +xgboostParams.learning_rate, max_depth: +xgboostParams.max_depth, n_estimators: +xgboostParams.n_estimators, min_child_weight: +xgboostParams.min_child_weight };
    }
    if (selectedModels.includes('Random Forest') || selectedModels.includes('Random Forest Regressor')) {
      const k = selectedModels.includes('Random Forest') ? 'Random Forest' : 'Random Forest Regressor';
      hp[k] = { n_estimators: +rfParams.n_estimators, max_depth: +rfParams.max_depth };
    }
    if (selectedModels.includes('Decision Tree') || selectedModels.includes('Decision Tree Regressor')) {
      const k = selectedModels.includes('Decision Tree') ? 'Decision Tree' : 'Decision Tree Regressor';
      hp[k] = { max_depth: +dtParams.max_depth, min_samples_split: +dtParams.min_samples_split };
    }
    if (selectedModels.some((m) => m.includes('SVM'))) {
      hp['Linear SVM'] = { C: +svmParams.C };
      hp['Kernel SVM'] = { C: +svmParams.C };
    }

    const payload = {
      dataset_id:      activeDataset.dataset_id,
      target_column:   finalTarget,
      task_type:       taskType,
      mode,
      feature_columns: selectedFeatures.length > 0 ? selectedFeatures : null,
      models:          mode === 'guided' ? null : selectedModels,
      cv_folds:        mode === 'guided' ? 10 : cvFolds,
      test_size:       mode === 'guided' ? 0.2 : testSplit / 100,
      scaler_type:     mode === 'guided' ? 'standard' : scalerType,
      hyperparameters: mode === 'guided' ? null : hp,
    };

    try {
      const resp = await api.createExperiment(payload);
      setActiveExpId(resp.experiment_id);
    } catch (err) {
      setRunning(false);
      setIsBenchmarking?.(false);
      setErrorMessage(err.message);
    }
  };

  const handleCancelBenchmark = async () => {
    if (!window.confirm('Stop this benchmark run? Incomplete models will be discarded.')) return;
    if (activeExpId) {
      try {
        await api.cancelExperiment(activeExpId);
      } catch (err) {
        console.warn('Cancel experiment error:', err);
      }
    }
    setRunning(false);
    setIsBenchmarking?.(false);
    setActiveExpId(null);
    setRunProgressMessage('');
    setErrorMessage('Benchmarking was stopped by user.');
  };

  const families = {};
  candidateModels.forEach((m) => { (families[m.family] = families[m.family] || []).push(m); });

  return (
    <div className="view-container anim-fade">
      <div className="studio-intro">
        <div className="studio-intro-copy">
          <div className="eyebrow">EXPERIMENT DESIGN / MODEL EVALUATION</div>
          <h1 className="page-title">Experiment Studio</h1>
          <p className="page-subtitle">
            Define the prediction task, select input columns, and choose how models are evaluated.
          </p>
        </div>
        <div className={`studio-dataset-status${activeDataset ? ' is-ready' : ''}`}>
          <span className="status-dot" />
          <span>
            <small>ACTIVE DATASET</small>
            <strong>{activeDataset?.name || activeDataset?.filename || 'Select a dataset first'}</strong>
          </span>
        </div>
      </div>

      {/* Lock Notification during Benchmarking */}
      {running && (
        <div
          className="alert alert-info"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            marginBottom: 20,
            padding: '14px 18px',
            border: '1.5px solid var(--info-border)',
            background: 'var(--info-bg)',
          }}
        >
          <Lock size={18} className="alert-icon" style={{ color: 'var(--info)' }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: 13.5, color: 'var(--text-main)', marginBottom: 2 }}>
              Benchmark Execution in Progress — Configuration Locked
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
              Parameters and model selections are frozen to prevent configuration mismatch and ensure valid holdout evaluation.
            </div>
          </div>
          <button
            className="btn btn-danger btn-sm"
            onClick={handleCancelBenchmark}
            style={{ flexShrink: 0, fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: 6 }}
            title="Cancel this benchmark run"
          >
            <Square size={13} style={{ fill: 'currentColor' }} />
            Stop Benchmark
          </button>
        </div>
      )}

      {/* Mode selector */}
      <div className="studio-mode-grid">
        {[
          { id: 'guided', icon: <Sparkles size={14} />, title: 'Guided Mode', sub: 'Smart defaults, auto model selection' },
          { id: 'expert', icon: <Sliders size={14} />,  title: 'Expert Mode',  sub: 'Full control over every parameter' },
        ].map(({ id, icon, title, sub }) => (
          <button
            key={id}
            type="button"
            aria-pressed={mode === id}
            disabled={running}
            onClick={() => setMode(id)}
            className={`studio-mode-choice${mode === id ? ' is-active' : ''}`}
            style={{
              opacity: running && mode !== id ? 0.45 : 1,
            }}
          >
            <div style={{
              width: 34, height: 34,
              background: mode === id ? 'var(--primary)' : 'var(--bg-subtle)',
              borderRadius: 'var(--radius-sm)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: mode === id ? '#fff' : 'var(--text-muted)',
            }}>{icon}</div>
            <div>
              <div style={{ fontSize: 13.5, fontWeight: 700, color: mode === id ? 'var(--primary)' : 'var(--text-main)' }}>
                {title}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{sub}</div>
            </div>
          </button>
        ))}
      </div>

      {/* Guided info card */}
      {mode === 'guided' && (
        <div
          className="card"
          style={{
            marginBottom: 20,
            border: '1px solid rgba(0, 229, 195, 0.25)',
            background: '#0d1524',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
          }}
        >
          <div className="card-body">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <Sparkles size={18} color="#00e5c3" />
              <div style={{ fontSize: 15, fontWeight: 700, color: '#f8fafc' }}>
                Guided Mode — Automated Best Practices
              </div>
            </div>
            <p style={{ fontSize: 13.5, color: '#94a3b8', lineHeight: 1.6, marginBottom: 14 }}>
              Guided mode selects models for <strong style={{ color: '#00e5c3' }}>{taskType.replace(/_/g, ' ')}</strong>, applies
              StandardScaler, reserves a 20% holdout, and requests 10-fold cross-validation. The backend may adapt validation
              to dataset constraints.
            </p>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {['10-Fold CV Request', '80/20 Holdout Split', 'StandardScaler', 'Train-Fold Preprocessing'].map((t) => (
                <span
                  key={t}
                  style={{
                    fontSize: 11.5,
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 600,
                    padding: '4px 10px',
                    borderRadius: 20,
                    background: 'rgba(0, 229, 195, 0.08)',
                    border: '1px solid rgba(0, 229, 195, 0.25)',
                    color: '#00e5c3',
                  }}
                >
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Expert mode controls */}
      {mode === 'expert' && (
        <>
          {/* Validation controls */}
          <div className="studio-validation-grid" style={{ marginBottom: 20 }}>
            {/* CV Folds */}
            <div className="card" style={{ marginBottom: 0 }}>
              <div className="card-body">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <div className="form-label" style={{ marginBottom: 0 }}>Cross-Validation Folds</div>
                  <code style={{ fontSize: 14, fontWeight: 700 }}>{cvFolds}</code>
                </div>
                <input
                  type="range" min="2" max="20" step="1" value={cvFolds}
                  disabled={running}
                  className="form-range" style={{ '--fill': `${((cvFolds - 2) / 18) * 100}%`, cursor: running ? 'not-allowed' : 'pointer', opacity: running ? 0.6 : 1 }}
                  onChange={(e) => setCvFolds(+e.target.value)}
                />
                <div className="form-hint">Adaptive fallback if dataset has &lt;25 rows</div>
              </div>
            </div>

            {/* Test Split */}
            <div className="card" style={{ marginBottom: 0 }}>
              <div className="card-body">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <div className="form-label" style={{ marginBottom: 0 }}>Holdout Test Split</div>
                  <code style={{ fontSize: 14, fontWeight: 700 }}>{testSplit}%</code>
                </div>
                <input
                  type="range" min="10" max="40" step="5" value={testSplit}
                  disabled={running}
                  className="form-range" style={{ '--fill': `${((testSplit - 10) / 30) * 100}%`, cursor: running ? 'not-allowed' : 'pointer', opacity: running ? 0.6 : 1 }}
                  onChange={(e) => setTestSplit(+e.target.value)}
                />
                <div className="form-hint">{testSplit}% test · {100 - testSplit}% train</div>
              </div>
            </div>

            {/* Scaler */}
            <div className="card" style={{ marginBottom: 0 }}>
              <div className="card-body">
                <div className="form-group" style={{ marginBottom: 8 }}>
                  <label className="form-label">Feature Scaler</label>
                  <select
                    className="form-select"
                    value={scalerType}
                    disabled={running}
                    style={{ cursor: running ? 'not-allowed' : 'pointer', opacity: running ? 0.6 : 1 }}
                    onChange={(e) => setScalerType(e.target.value)}
                  >
                    <option value="standard">StandardScaler — zero mean, unit variance</option>
                    <option value="minmax">MinMaxScaler — bounded [0, 1]</option>
                    <option value="robust">RobustScaler — median/IQR, outlier-safe</option>
                    <option value="none">None — raw numerical values</option>
                  </select>
                </div>
                <div className="form-hint">Preprocessing is fitted within training folds.</div>
              </div>
            </div>
          </div>

          {/* Model Catalog */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Model Catalog — {candidateModels.length} Algorithms Available</div>
                <div className="card-subtitle">{selectedModels.length} selected for benchmarking</div>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="btn btn-ghost btn-sm"
                  disabled={running}
                  style={running ? { cursor: 'not-allowed', opacity: 0.5 } : undefined}
                  onClick={() => !running && setSelectedModels(candidateModels.map((m) => m.name))}
                >
                  Select All
                </button>
                <button
                  className="btn btn-ghost btn-sm"
                  disabled={running}
                  style={running ? { cursor: 'not-allowed', opacity: 0.5 } : undefined}
                  onClick={() => !running && setSelectedModels([candidateModels[0].name])}
                >
                  Reset
                </button>
              </div>
            </div>
            <div className="card-body" style={{ paddingTop: 16 }}>
              {Object.entries(families).map(([famName, models]) => (
                <div key={famName} style={{ marginBottom: 18 }}>
                  <div className="section-label">
                    <span style={{ color: FAMILY_COLORS[famName] || 'var(--text-muted)' }}>■</span>
                    {famName}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: 8 }}>
                    {models.map((m) => {
                      const checked = selectedModels.includes(m.name);
                      return (
                        <button
                          key={m.name}
                          type="button"
                          aria-pressed={checked}
                          disabled={running}
                          onClick={() => toggleModel(m.name)}
                          style={{
                            display: 'flex', alignItems: 'center', gap: 9, padding: '9px 12px',
                            borderRadius: 'var(--radius-sm)',
                            border: `1.5px solid ${checked ? 'var(--primary)' : 'var(--border-color)'}`,
                            background: checked ? 'var(--primary-light)' : 'var(--bg-subtle)',
                            cursor: running ? 'not-allowed' : 'pointer',
                            opacity: running ? 0.6 : 1,
                            transition: 'all var(--duration-fast)',
                            color: 'var(--text-main)',
                            textAlign: 'left',
                            fontFamily: 'var(--font-sans)',
                          }}
                        >
                          {checked
                            ? <CheckSquare size={15} color="var(--primary)" style={{ flexShrink: 0 }} />
                            : <Square size={15} color="var(--text-subtle)" style={{ flexShrink: 0 }} />}
                          <span style={{ fontSize: 13, fontWeight: 600, color: checked ? 'var(--primary)' : 'var(--text-main)' }}>
                            {m.name}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Hyperparameters */}
          <div className="card">
            <div
              className="card-header"
              style={{ cursor: running ? 'not-allowed' : 'pointer' }}
              onClick={() => !running && setShowHyperparams((v) => !v)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Settings2 size={16} color="var(--primary)" />
                <div>
                  <div className="card-title">Hyperparameter Configuration</div>
                  <div className="card-subtitle">Override defaults for selected models</div>
                </div>
              </div>
              {showHyperparams ? <ChevronUp size={16} color="var(--text-muted)" /> : <ChevronDown size={16} color="var(--text-muted)" />}
            </div>

            {showHyperparams && (
              <div className="card-body">
                <div className="studio-hyperparameter-grid">
                  {/* XGBoost */}
                  <div style={{ padding: 16, borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', background: 'var(--bg-subtle)' }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-main)', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#dc2626', display: 'inline-block' }} />
                      XGBoost
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                      <HyperParamInput label="Learning Rate (eta)" hint="0.01–1.0" value={xgboostParams.learning_rate} disabled={running} step="0.01" min="0.01" max="1" onChange={(e) => setXgboostParams({ ...xgboostParams, learning_rate: e.target.value })} />
                      <HyperParamInput label="Max Depth" hint="2–20" value={xgboostParams.max_depth} disabled={running} min="2" max="20" onChange={(e) => setXgboostParams({ ...xgboostParams, max_depth: e.target.value })} />
                      <HyperParamInput label="Estimators" hint="10–1000" value={xgboostParams.n_estimators} disabled={running} min="10" max="1000" step="10" onChange={(e) => setXgboostParams({ ...xgboostParams, n_estimators: e.target.value })} />
                      <HyperParamInput label="Min Child Weight" hint="1–20" value={xgboostParams.min_child_weight} disabled={running} min="1" max="20" onChange={(e) => setXgboostParams({ ...xgboostParams, min_child_weight: e.target.value })} />
                    </div>
                  </div>

                  {/* Tree & SVM */}
                  <div style={{ padding: 16, borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', background: 'var(--bg-subtle)' }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-main)', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#059669', display: 'inline-block' }} />
                      Random Forest · Decision Tree · SVM
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                      <HyperParamInput label="RF Estimators" hint="10–500" value={rfParams.n_estimators} disabled={running} min="10" max="500" step="10" onChange={(e) => setRfParams({ ...rfParams, n_estimators: e.target.value })} />
                      <HyperParamInput label="RF Max Depth" hint="2–50" value={rfParams.max_depth} disabled={running} min="2" max="50" onChange={(e) => setRfParams({ ...rfParams, max_depth: e.target.value })} />
                      <HyperParamInput label="DT Max Depth" hint="2–30" value={dtParams.max_depth} disabled={running} min="2" max="30" onChange={(e) => setDtParams({ ...dtParams, max_depth: e.target.value })} />
                      <HyperParamInput label="SVM C (regularization)" hint="0.01–100" value={svmParams.C} disabled={running} step="0.1" min="0.01" max="100" onChange={(e) => setSvmParams({ C: e.target.value })} />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* Task Type and Target Column selector */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-body" style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
          <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: 200 }}>
            <label className="form-label">Task Type</label>
            <select
              className="form-select"
              value={taskType}
              disabled={running}
              style={{ cursor: running ? 'not-allowed' : 'pointer', opacity: running ? 0.6 : 1 }}
              onChange={(e) => setTaskType(e.target.value)}
            >
              <option value="binary_classification">Binary Classification</option>
              <option value="multiclass_classification">Multiclass Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
          <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: 220 }}>
            <label className="form-label">Target Column (Prediction Goal)</label>
            <select
              className="form-select"
              value={selectedTarget || ''}
              disabled={running}
              style={{ cursor: running ? 'not-allowed' : 'pointer', opacity: running ? 0.6 : 1 }}
              onChange={handleTargetChange}
            >
              {candidateTargets.length > 0 ? (
                candidateTargets.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))
              ) : selectedTarget ? (
                <option value={selectedTarget}>{selectedTarget}</option>
              ) : (
                <option value="">Select Target Column...</option>
              )}
            </select>
          </div>
          {activeDataset && (
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Active Dataset</label>
              <div style={{ padding: '9px 0', fontSize: 13.5, fontWeight: 600, color: 'var(--primary)' }}>
                {activeDataset.name || activeDataset.filename}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input Feature Columns Selection */}
      {candidateTargets.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-header">
            <div>
              <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Layers size={16} color="var(--primary)" />
                <span>Input Feature Columns ({selectedFeatures.length} of {candidateTargets.filter((c) => c !== selectedTarget).length} Selected)</span>
              </div>
              <div className="card-subtitle">
                Choose which columns to include as predictive features. Uncheck leaked proxies, ground-truth outcomes, or unique IDs.
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                disabled={running}
                style={running ? { cursor: 'not-allowed', opacity: 0.5 } : undefined}
                onClick={() => !running && setSelectedFeatures(candidateTargets.filter((c) => c !== selectedTarget))}
              >
                Select All
              </button>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                disabled={running}
                style={running ? { cursor: 'not-allowed', opacity: 0.5 } : undefined}
                onClick={() => !running && setSelectedFeatures([])}
              >
                Deselect All
              </button>
              {leakageColumns.length > 0 && (
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  disabled={running}
                  style={running ? { cursor: 'not-allowed', opacity: 0.5, color: '#dc2626' } : { color: '#dc2626' }}
                  onClick={() => !running && setSelectedFeatures((prev) => prev.filter((c) => !leakageColumns.includes(c)))}
                  title="Exclude all columns flagged by data quality & leakage audit"
                >
                  Exclude Flagged ({leakageColumns.length})
                </button>
              )}
            </div>
          </div>
          <div className="card-body" style={{ paddingTop: 14 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 10 }}>
              {candidateTargets
                .filter((col) => col !== selectedTarget)
                .map((col) => {
                  const isChecked = selectedFeatures.includes(col);
                  const isLeak = leakageColumns.includes(col);
                  return (
                    <button
                      key={col}
                      type="button"
                      aria-pressed={isChecked}
                      disabled={running}
                      onClick={() => toggleFeature(col)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '10px 14px',
                        borderRadius: 'var(--radius-sm)',
                        border: `1.5px solid ${
                          isLeak && isChecked
                            ? '#f59e0b'
                            : isChecked
                            ? 'var(--primary)'
                            : 'var(--border-color)'
                        }`,
                        background: isChecked ? (isLeak ? 'rgba(245,158,11,0.12)' : 'var(--primary-light)') : 'var(--bg-subtle)',
                        cursor: running ? 'not-allowed' : 'pointer',
                        opacity: running ? 0.6 : 1,
                        transition: 'all var(--duration-fast)',
                        color: 'var(--text-main)',
                        textAlign: 'left',
                        fontFamily: 'var(--font-sans)',
                      }}
                      title={isLeak ? `Warning: ${col} was flagged for potential target leakage` : col}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 9, minWidth: 0 }}>
                        {isChecked ? (
                          <CheckSquare size={16} color={isLeak ? '#d97706' : 'var(--primary)'} style={{ flexShrink: 0 }} />
                        ) : (
                          <Square size={16} color="var(--text-subtle)" style={{ flexShrink: 0 }} />
                        )}
                        <span
                          style={{
                            fontSize: 13,
                            fontWeight: 600,
                            color: isChecked ? 'var(--text-main)' : 'var(--text-muted)',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {col}
                        </span>
                      </div>
                      {isLeak && (
                        <span
                          className="badge"
                          style={{
                            fontSize: 10,
                            padding: '2px 6px',
                            background: 'var(--warning-bg)',
                            color: 'var(--warning-text)',
                            border: '1px solid var(--warning-border)',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 3,
                            flexShrink: 0,
                          }}
                        >
                          <AlertTriangle size={10} /> Leakage
                        </span>
                      )}
                    </button>
                  );
                })}
            </div>
            {selectedFeatures.length === 0 && (
              <div className="alert alert-danger" style={{ marginTop: 12, padding: '8px 12px', fontSize: 12 }}>
                <AlertCircle size={14} className="alert-icon" />
                <span>At least one input feature column must be selected to benchmark.</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Launch bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'var(--bg-card)', border: '1.5px solid var(--primary-mid)',
        borderRadius: 'var(--radius-md)', padding: '20px 24px', boxShadow: 'var(--shadow-blue)',
        gap: 20,
      }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-main)', marginBottom: 4 }}>
            {mode === 'guided'
              ? 'Run Automated Guided Benchmark'
              : `Benchmark ${selectedModels.length} Selected Estimator${selectedModels.length !== 1 ? 's' : ''}`}
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
            Computes holdout metrics, cross-validation variance, and SHAP feature attribution
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {running && (
            <button
              className="btn btn-danger btn-lg"
              onClick={handleCancelBenchmark}
              style={{ flexShrink: 0, display: 'inline-flex', alignItems: 'center', gap: 8 }}
              title="Cancel and stop current benchmark"
            >
              <Square size={14} style={{ fill: 'currentColor' }} />
              <span>Stop Benchmark</span>
            </button>
          )}
          <button
            className="btn btn-primary btn-lg"
            onClick={handleLaunch}
            disabled={running || !activeDataset || maintenanceMode}
            style={{
              flexShrink: 0,
              cursor: maintenanceMode ? 'not-allowed' : undefined,
              opacity: maintenanceMode ? 0.6 : 1,
            }}
            title={maintenanceMode ? 'Training is temporarily disabled while platform maintenance is active.' : undefined}
          >
            {running ? <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> : <Play size={16} />}
            {running ? 'Benchmarking…' : maintenanceMode ? 'Maintenance Active' : 'Launch Experiment'}
          </button>
        </div>
      </div>

      {running && (
        <div className="alert alert-info" style={{ marginTop: 14 }}>
          <div className="spinner alert-icon" />
          <span>{runProgressMessage}</span>
        </div>
      )}

      {errorMessage && (
        <div className="alert alert-danger" style={{ marginTop: 14 }}>
          <AlertCircle size={15} className="alert-icon" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
}
