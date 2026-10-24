import { History, Award, Trash2, ArrowRight, Clock, TrendingUp, AlertCircle, Loader } from 'lucide-react';
import { api } from '../api';

export default function HistoryView({ experiments, onSelectExperiment, onExperimentDeleted }) {
  const handleDelete = async (expId) => {
    if (!window.confirm('Delete this experiment and all associated artifact bundles? This cannot be undone.')) return;
    try {
      await api.deleteExperiment(expId);
      onExperimentDeleted(expId);
    } catch (err) {
      alert(`Deletion failed: ${err.message}`);
    }
  };

  const completed = experiments.filter((e) => e.status === 'completed').length;
  const failed    = experiments.filter((e) => e.status === 'failed').length;

  return (
    <div className="view-container anim-fade">
      <div className="page-header">
        <div className="page-title">Run History</div>
        <div className="page-subtitle">Audit trail of all benchmarked experiments and associated model bundles</div>
      </div>

      {/* Summary stats */}
      <div className="stat-grid" style={{ marginBottom: 20 }}>
        <div className="stat-card blue">
          <div className="stat-label">Total Runs</div>
          <div className="stat-value">{experiments.length}</div>
          <div className="stat-meta">All time</div>
        </div>
        <div className="stat-card green">
          <div className="stat-label">Completed</div>
          <div className="stat-value">{completed}</div>
          <div className="stat-meta">Successfully benchmarked</div>
        </div>
        <div className="stat-card danger">
          <div className="stat-label">Failed</div>
          <div className="stat-value">{failed}</div>
          <div className="stat-meta">Inspect logs for details</div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">All Experiments</div>
            <div className="card-subtitle">Click Results to inspect leaderboard and feature attribution</div>
          </div>
          <span className="badge badge-neutral">{experiments.length} runs</span>
        </div>

        {experiments.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon"><History size={24} /></div>
            <div className="empty-title">No experiment runs yet</div>
            <div className="empty-desc">Launch a benchmark in Experiment Studio to see it here.</div>
          </div>
        ) : (
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Winning Model</th>
                  <th>Dataset</th>
                  <th>Target</th>
                  <th>Task Type</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>Timestamp</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp) => {
                  const StatusIcon = exp.status === 'completed' ? TrendingUp
                    : exp.status === 'failed' ? AlertCircle
                    : Loader;

                  return (
                    <tr key={exp.experiment_id}>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                          {exp.best_model && <Award size={13} color="#d97706" />}
                          <strong style={{ fontSize: 13.5 }}>{exp.best_model || '—'}</strong>
                        </div>
                      </td>
                      <td style={{ color: 'var(--text-muted)', fontSize: 13 }}>{exp.dataset_name}</td>
                      <td>
                        <code style={{ fontSize: 12 }}>{exp.target_column}</code>
                      </td>
                      <td>
                        <span className="badge badge-info" style={{ textTransform: 'capitalize', fontSize: 11 }}>
                          {exp.task_type?.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td>
                        <span className="badge badge-neutral" style={{ textTransform: 'capitalize', fontSize: 11 }}>
                          {exp.mode || 'guided'}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${
                          exp.status === 'completed' ? 'badge-success'
                          : exp.status === 'running'   ? 'badge-warning'
                          : 'badge-danger'
                        }`} style={{ fontSize: 11 }}>
                          {exp.status}
                        </span>
                      </td>
                      <td style={{ color: 'var(--text-muted)', fontSize: 12, whiteSpace: 'nowrap' }}>
                        {exp.created_at ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                            <Clock size={11} />
                            {new Date(exp.created_at).toLocaleString()}
                          </div>
                        ) : '—'}
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 7, justifyContent: 'flex-end' }}>
                          {exp.status === 'completed' && (
                            <button
                              className="btn btn-secondary btn-sm"
                              onClick={() => onSelectExperiment(exp.experiment_id)}
                            >
                              Results <ArrowRight size={12} />
                            </button>
                          )}
                          <button
                            className="btn btn-danger btn-sm btn-icon"
                            onClick={() => handleDelete(exp.experiment_id)}
                            title="Delete run & artifacts"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
