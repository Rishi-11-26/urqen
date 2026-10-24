import React, { useState } from 'react';
import {
  Database, ShieldCheck, Sparkles, Cpu, Award, FileText,
  ArrowRight, CheckCircle2, AlertTriangle, Layers, Activity,
  Lock, Zap, Check
} from 'lucide-react';
import Footer from '../components/Footer';

const PIPELINE_STEPS = [
  {
    step: '01',
    title: 'Upload & profile',
    icon: Database,
    color: '#00e5c3',
    tag: 'File input',
    summary: 'Upload a CSV or Excel file. URQEN parses the table and provides a dataset profile.',
    details: [
      'Supported formats are CSV, XLSX, and XLS.',
      'Review the parsed columns and dataset summaries before continuing.',
      'Choose which uploaded dataset to use in an experiment.'
    ]
  },
  {
    step: '02',
    title: 'Quality & leakage review',
    icon: ShieldCheck,
    color: '#38bdf8',
    tag: 'Review signals',
    summary: 'Inspect data-quality findings and leakage indicators generated for the selected dataset.',
    details: [
      'Review findings, evidence, and suggested actions in Dataset Lab.',
      'Use the audit to identify potential issues that deserve further inspection.',
      'An automated audit is a guide; it does not certify a dataset or guarantee leakage-free results.'
    ]
  },
  {
    step: '03',
    title: 'Configure an experiment',
    icon: Layers,
    color: '#6366f1',
    tag: 'Experiment setup',
    summary: 'Choose a target, task, and input features, then configure the experiment in Studio.',
    details: [
      'Select the column you want the model to predict.',
      'Choose which available columns to include as input features.',
      'Preprocessing and model options depend on the selected task and configuration.'
    ]
  },
  {
    step: '04',
    title: 'Run a benchmark',
    icon: Cpu,
    color: '#a855f7',
    tag: 'Model evaluation',
    summary: 'Run the configured benchmark and evaluate the candidate models supported for that task.',
    details: [
      'Candidate models and evaluation details depend on the selected task and settings.',
      'The backend reports the metrics available for the completed run.',
      'Runtime and results depend on the dataset, configuration, and deployment resources.'
    ]
  },
  {
    step: '05',
    title: 'Compare results',
    icon: Award,
    color: '#f59e0b',
    tag: 'Measured outcomes',
    summary: 'Review the model comparison and evaluation results returned for your experiment.',
    details: [
      'Inspect the reported scores and validation summaries where available.',
      'Use Results & Inference to explore the selected experiment further.',
      'Validation scores are estimates and are not guarantees of future performance.'
    ]
  },
  {
    step: '06',
    title: 'Review and export',
    icon: FileText,
    color: '#10b981',
    tag: 'Experiment artifacts',
    summary: 'Review available diagnostics and download artifacts produced by the experiment.',
    details: [
      'Available diagnostics depend on the experiment task and results.',
      'Download the reports or model artifacts provided for the run.',
      'Review and independently validate any model before using it outside the studio.'
    ]
  }
];

const COMPARISON_ROWS = [
  {
    feature: 'Data quality',
    urqen: 'Dataset Lab presents quality findings, evidence, and recommendations.',
    typical: 'Findings are indicators to review; they do not certify data quality or rule out leakage.'
  },
  {
    feature: 'Model evaluation',
    urqen: 'Review the evaluation metrics and validation summaries returned for a run.',
    typical: 'Scores describe the evaluated data and settings; they cannot guarantee future performance.'
  },
  {
    feature: 'Experiment setup',
    urqen: 'Choose a target, task, and input features in Experiment Studio.',
    typical: 'Available models and preprocessing depend on task, data, and configuration.'
  },
  {
    feature: 'Diagnostics',
    urqen: 'Results provides diagnostics and artifacts supported by the completed run.',
    typical: 'The available explanations vary by task and model.'
  },
  {
    feature: 'Feature selection',
    urqen: 'Choose which eligible columns to include in the benchmark.',
    typical: 'Review the selected inputs for relevance and possible leakage.'
  }
];

export default function HowItWorksView({ onNavigate, onOpenLegal }) {
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const activeStep = PIPELINE_STEPS[activeStepIndex];

  return (
    <div className="view-container anim-fade">
      {/* ── Hero Banner ────────────────────────────────────────── */}
      <div
        className="card"
        style={{
          marginBottom: 28,
          background: 'linear-gradient(135deg, rgba(0, 229, 195, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%)',
          border: '1px solid rgba(0, 229, 195, 0.25)',
          borderRadius: 'var(--radius-lg)',
          padding: '32px 36px',
          boxShadow: '0 12px 32px rgba(0, 0, 0, 0.5)',
        }}
      >
        <div style={{ maxWidth: 760 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, marginBottom: 12 }} className="badge badge-primary">
            <Sparkles size={12} />
            <span>Architecture & System Blueprint</span>
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em', marginBottom: 12, lineHeight: 1.25 }}>
            How URQEN Supports Transparent Machine Learning
          </h1>
          <p style={{ fontSize: 14.5, color: 'var(--text-muted)', lineHeight: 1.65, margin: 0 }}>
            Explore the current workflow from tabular data inspection through model evaluation and export.
            Audit findings are review signals, and model scores are estimates based on the selected data and configuration.
          </p>

          <div style={{ display: 'flex', gap: 10, marginTop: 20, flexWrap: 'wrap' }}>
            <button
              className="btn btn-primary"
              onClick={() => onNavigate?.('dataset-lab')}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}
            >
              <span>Explore Dataset Lab</span>
              <ArrowRight size={14} />
            </button>
            <button
              className="btn btn-neutral"
              onClick={() => onNavigate?.('experiment-studio')}
            >
              Configure an Experiment
            </button>
          </div>
        </div>
      </div>

      {/* ── 6-Step Visual Lifecycle Grid ──────────────────────── */}
      <div style={{ marginBottom: 28 }}>
        <div className="section-title" style={{ fontSize: 18, marginBottom: 6 }}>
          The Experiment Workflow
        </div>
        <div className="section-desc" style={{ marginBottom: 16 }}>
          Select a step to see what the app does and what to review.
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10, marginBottom: 18 }}>
          {PIPELINE_STEPS.map((s, idx) => {
            const Icon = s.icon;
            const isSelected = activeStepIndex === idx;
            return (
              <button
                type="button"
                key={s.step}
                onClick={() => setActiveStepIndex(idx)}
                aria-pressed={isSelected}
                style={{
                  padding: '14px 16px',
                  borderRadius: 'var(--radius-md)',
                  border: `2px solid ${isSelected ? s.color : 'var(--border-color)'}`,
                  background: isSelected ? 'var(--bg-card)' : 'var(--bg-subtle)',
                  cursor: 'pointer',
                  transition: 'all var(--duration-fast)',
                  boxShadow: isSelected ? '0 4px 12px rgba(0,0,0,0.06)' : 'none',
                  textAlign: 'left',
                  fontFamily: 'inherit',
                  width: '100%',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontSize: 11, fontWeight: 800, color: s.color, letterSpacing: '0.05em' }}>
                    STEP {s.step}
                  </span>
                  <div
                    style={{
                      width: 26, height: 26, borderRadius: 6,
                      background: isSelected ? s.color : 'var(--border-color)',
                      color: isSelected ? '#fff' : 'var(--text-muted)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}
                  >
                    <Icon size={14} />
                  </div>
                </div>
                <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-main)', lineHeight: 1.3 }}>
                  {s.title}
                </div>
              </button>
            );
          })}
        </div>

        {/* Selected Step Detail Inspector */}
        <div className="card" style={{ borderLeft: `4px solid ${activeStep.color}` }}>
          <div className="card-body">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <span className="badge" style={{ background: `${activeStep.color}15`, color: activeStep.color, fontWeight: 700 }}>
                {activeStep.tag}
              </span>
              <h3 style={{ fontSize: 17, fontWeight: 700, margin: 0, color: 'var(--text-main)' }}>
                Step {activeStep.step}: {activeStep.title}
              </h3>
            </div>
            <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: 16 }}>
              {activeStep.summary}
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
              {activeStep.details.map((item, dIdx) => (
                <div
                  key={dIdx}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 10,
                    padding: '12px 14px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-subtle)',
                    border: '1px solid var(--border-color)',
                  }}
                >
                  <CheckCircle2 size={16} color={activeStep.color} style={{ flexShrink: 0, marginTop: 2 }} />
                  <span style={{ fontSize: 13, color: 'var(--text-main)', lineHeight: 1.5 }}>
                    {item}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── URQEN workflow comparison table ───────────── */}
      <div className="card" style={{ marginBottom: 28 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Outputs and limitations</div>
            <div className="card-subtitle">What URQEN shows and what you should still verify</div>
          </div>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '22%' }}>Stage</th>
                <th style={{ width: '39%' }}>In URQEN</th>
                <th style={{ width: '39%' }}>Context</th>
              </tr>
            </thead>
            <tbody>
              {COMPARISON_ROWS.map((row, idx) => (
                <tr key={idx}>
                  <td style={{ fontWeight: 700, color: 'var(--text-main)', fontSize: 13 }}>
                    {row.feature}
                  </td>
                  <td style={{ color: 'var(--success-dark, #065f46)', fontSize: 13, lineHeight: 1.5 }}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                      <Check size={14} color="var(--success)" style={{ flexShrink: 0, marginTop: 3 }} />
                      <span>{row.urqen}</span>
                    </div>
                  </td>
                  <td style={{ color: 'var(--text-muted)', fontSize: 12.5, lineHeight: 1.5 }}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                      <AlertTriangle size={14} color="#d97706" style={{ flexShrink: 0, marginTop: 3 }} />
                      <span>{row.typical}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Practical notes ───────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16, marginBottom: 32 }}>
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-body">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <Lock size={18} color="var(--primary)" />
              <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-main)' }}>Experiment state</div>
            </div>
            <p style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
              The Studio displays experiment progress and disables configuration controls while a run is active. Worker isolation depends on the backend deployment.
            </p>
          </div>
        </div>

        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-body">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <Zap size={18} color="#f59e0b" />
              <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-main)' }}>Runtime expectations</div>
            </div>
            <p style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
              Runtime depends on the dataset, candidate models, selected evaluation settings, and available backend resources. There is no fixed runtime guarantee.
            </p>
          </div>
        </div>

        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-body">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <Activity size={18} color="var(--success)" />
              <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-main)' }}>Reproducible Runs</div>
            </div>
            <p style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
              Some model and split operations use fixed random states. Matching scores across different data, package versions, or environments are not guaranteed.
            </p>
          </div>
        </div>
      </div>

      {/* ── Footer ─────────────────────────────────────────── */}
      <Footer onOpenLegal={onOpenLegal} />
    </div>
  );
}
