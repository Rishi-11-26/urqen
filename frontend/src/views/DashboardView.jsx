import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Database,
  FlaskConical,
  Upload,
} from 'lucide-react';

const WORKFLOW = [
  { number: '01', label: 'Bring in data', detail: 'Upload a CSV or Excel file', tab: 'dataset-lab', icon: Upload },
  { number: '02', label: 'Inspect', detail: 'Review its profile and quality', tab: 'dataset-lab', icon: Database },
  { number: '03', label: 'Configure', detail: 'Choose a target and task', tab: 'experiment-studio', icon: FlaskConical },
  { number: '04', label: 'Evaluate', detail: 'Compare results and diagnostics', tab: 'history', icon: BarChart3 },
];

export default function DashboardView({ datasets, experiments, onNavigate, onSelectDataset, onSelectExperiment }) {
  const completedExps = experiments.filter((experiment) => experiment.status === 'completed');
  const runningExps = experiments.filter((experiment) => experiment.status === 'running');
  const championCount = completedExps.filter((experiment) => experiment.best_model).length;

  return (
    <div className="view-container dashboard-view anim-fade">
      <section className="dashboard-hero">
        <div className="dashboard-hero-copy">
          <div className="dashboard-kicker">
            <span className="dashboard-live-dot" />
            YOUR TABULAR ML WORKSPACE
          </div>
          <h1>
            Make your data
            <br />
            <span>make its case.</span>
          </h1>
          <p>
            Move from raw tables to considered experiments. Inspect the data, choose what to test,
            then review what the results actually show.
          </p>
          <div className="dashboard-actions">
            <button className="btn btn-primary dashboard-primary-action" onClick={() => onNavigate('dataset-lab')}>
              <Upload size={16} />
              Open Dataset Lab
              <ArrowUpRight size={15} />
            </button>
            <button className="dashboard-text-action" onClick={() => onNavigate('experiment-studio')}>
              Go to Experiment Studio <ArrowRight size={15} />
            </button>
          </div>
        </div>

        <div className="dashboard-hero-visual" aria-label="Workflow overview: data, inspection, configuration, evaluation">
          <div className="dashboard-visual-topline">
            <span>FROM TABLE TO TEST</span>
            <span>04 STEPS</span>
          </div>
          <div className="dashboard-signal" aria-hidden="true">
            <div className="dashboard-signal-grid" />
            <svg viewBox="0 0 480 150" preserveAspectRatio="none">
              <path className="dashboard-signal-guide" d="M0 112H480" />
              <path className="dashboard-signal-line" d="M0 112 C40 112 40 64 84 64 S130 105 176 105 S219 40 270 40 S317 84 357 84 S414 22 480 22" />
              <circle className="dashboard-signal-end" cx="480" cy="22" r="4" />
            </svg>
            <span className="dashboard-signal-caption">A workflow you can follow</span>
          </div>
          <div className="dashboard-visual-stages">
            {WORKFLOW.map(({ number, label }) => (
              <div className="dashboard-visual-stage" key={number}>
                <span>{number}</span>
                <strong>{label}</strong>
              </div>
            ))}
          </div>
          <div className="dashboard-visual-note">
            <span className="dashboard-live-dot" />
            Explore each step at your own pace
          </div>
        </div>

        <div className="dashboard-hero-index" aria-hidden="true">01 / 06</div>
      </section>

      <section className="dashboard-metrics" aria-label="Workspace summary">
        <div className="dashboard-metric">
          <div className="dashboard-metric-label"><Database size={14} /> DATASETS</div>
          <strong>{datasets.length.toString().padStart(2, '0')}</strong>
          <span>{datasets.length === 1 ? 'ready to inspect' : 'in your workspace'}</span>
        </div>
        <div className="dashboard-metric">
          <div className="dashboard-metric-label"><FlaskConical size={14} /> EXPERIMENTS</div>
          <strong>{experiments.length.toString().padStart(2, '0')}</strong>
          <span>{runningExps.length} currently running</span>
        </div>
        <div className="dashboard-metric">
          <div className="dashboard-metric-label"><BarChart3 size={14} /> COMPLETED</div>
          <strong>{completedExps.length.toString().padStart(2, '0')}</strong>
          <span>results available to review</span>
        </div>
        <div className="dashboard-metric dashboard-metric-accent">
          <div className="dashboard-metric-label"><Activity size={14} /> BEST MODELS</div>
          <strong>{championCount.toString().padStart(2, '0')}</strong>
          <span>from completed experiments</span>
        </div>
      </section>

      <section className="dashboard-workflow-section">
        <div className="dashboard-section-heading">
          <div>
            <div className="dashboard-section-index">THE WORKFLOW / 01—04</div>
            <h2>One clear step at a time.</h2>
          </div>
          <p>Move through the existing URQEN pipeline, from upload to evaluation.</p>
        </div>
        <div className="dashboard-workflow">
          {WORKFLOW.map(({ number, label, detail, tab, icon: Icon }) => (
            <button className="dashboard-workflow-step" key={number} onClick={() => onNavigate(tab)}>
              <span className="dashboard-workflow-index">{number}</span>
              <span className="dashboard-workflow-icon"><Icon size={19} strokeWidth={1.6} /></span>
              <span className="dashboard-workflow-copy">
                <strong>{label}</strong>
                <small>{detail}</small>
              </span>
              <ArrowUpRight className="dashboard-workflow-arrow" size={16} />
            </button>
          ))}
        </div>
      </section>

      <section className="dashboard-recent-section">
        <div className="dashboard-section-heading dashboard-recent-heading">
          <div>
            <div className="dashboard-section-index">YOUR WORK / RECENT</div>
            <h2>Pick up where you left off.</h2>
          </div>
        </div>
        <div className="dashboard-recent-grid">
          <article className="dashboard-recent-panel">
            <header className="dashboard-panel-heading">
              <div>
                <span className="dashboard-panel-overline">01 / DATA</span>
                <h3>Datasets</h3>
              </div>
              <button className="dashboard-panel-link" onClick={() => onNavigate('dataset-lab')}>
                All datasets <ArrowRight size={14} />
              </button>
            </header>
            {datasets.length === 0 ? (
              <div className="dashboard-empty">
                <Database size={20} />
                <div>
                  <strong>No datasets yet</strong>
                  <span>Upload a tabular file to begin exploring your data.</span>
                </div>
                <button className="dashboard-empty-action" onClick={() => onNavigate('dataset-lab')} aria-label="Open Dataset Lab">
                  <ArrowUpRight size={17} />
                </button>
              </div>
            ) : (
              <div className="dashboard-recent-list">
                {datasets.slice(0, 5).map((dataset) => (
                  <button
                    className="dashboard-recent-item"
                    key={dataset.dataset_id}
                    onClick={() => onSelectDataset(dataset)}
                  >
                    <span className="dashboard-recent-icon"><Database size={16} /></span>
                    <span className="dashboard-recent-copy">
                      <strong>{dataset.name || dataset.filename || 'Dataset'}</strong>
                      <small>
                        {(dataset.row_count ?? dataset.n_rows ?? 0).toLocaleString()} rows
                        <span>·</span>
                        {dataset.column_count ?? dataset.n_columns ?? 0} columns
                        <span>·</span>
                        {(dataset.file_type || dataset.file_format || 'csv').toUpperCase()}
                      </small>
                    </span>
                    <ArrowUpRight className="dashboard-recent-arrow" size={15} />
                  </button>
                ))}
                {datasets.length > 5 && (
                  <div className="dashboard-list-more">+{datasets.length - 5} more datasets</div>
                )}
              </div>
            )}
          </article>

          <article className="dashboard-recent-panel">
            <header className="dashboard-panel-heading">
              <div>
                <span className="dashboard-panel-overline">02 / EVIDENCE</span>
                <h3>Experiments</h3>
              </div>
              <button className="dashboard-panel-link" onClick={() => onNavigate('history')}>
                Run history <ArrowRight size={14} />
              </button>
            </header>
            {experiments.length === 0 ? (
              <div className="dashboard-empty">
                <FlaskConical size={20} />
                <div>
                  <strong>No experiments yet</strong>
                  <span>Choose a dataset and configure your first experiment.</span>
                </div>
                <button className="dashboard-empty-action" onClick={() => onNavigate('experiment-studio')} aria-label="Open Experiment Studio">
                  <ArrowUpRight size={17} />
                </button>
              </div>
            ) : (
              <div className="dashboard-recent-list">
                {experiments.slice(0, 5).map((experiment) => (
                  <button
                    className="dashboard-recent-item"
                    key={experiment.experiment_id}
                    onClick={() => onSelectExperiment(experiment.experiment_id)}
                  >
                    <span className={`dashboard-recent-icon${experiment.status === 'completed' ? ' is-complete' : ''}`}>
                      <Activity size={16} />
                    </span>
                    <span className="dashboard-recent-copy">
                      <strong>{experiment.dataset_name || 'Experiment'} <span className="dashboard-target-arrow">→</span> {experiment.target_column || 'Target'}</strong>
                      <small>
                        {experiment.task_type || 'Task not specified'}
                        <span>·</span>
                        {experiment.best_model ? `Best model: ${experiment.best_model}` : experiment.status}
                      </small>
                    </span>
                    <span className={`dashboard-status${experiment.status === 'completed' ? ' is-complete' : ''}`}>
                      {experiment.status}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </article>
        </div>
      </section>
    </div>
  );
}
