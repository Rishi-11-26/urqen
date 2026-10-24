import { Database, FlaskConical, BarChart3, History, ChevronRight, Compass, Shield } from 'lucide-react';
import UrqenLogo from './UrqenLogo';

const NAV_ITEMS = [
  { id: 'dashboard',          icon: BarChart3,    label: 'Overview' },
  { id: 'dataset-lab',        icon: Database,      label: 'Dataset Lab' },
  { id: 'experiment-studio',  icon: FlaskConical,  label: 'Experiment Studio' },
  { id: 'results',            icon: BarChart3,     label: 'Results & Inference' },
  { id: 'history',            icon: History,       label: 'Run History' },
  { id: 'how-it-works',       icon: Compass,       label: 'How It Works' },
];

export default function Sidebar({ currentTab, setCurrentTab, storageQuota, backendOnline, hasActiveExperiment, isAdmin }) {
  const usedBytes  = Number(storageQuota?.total_used_bytes) || 0;
  const limitBytes = Number(storageQuota?.quota_limit_bytes) || 524288000;
  const pct        = Math.min(100, Math.round((usedBytes / limitBytes) * 100));
  const usedMB     = usedBytes / 1048576;
  const limitMB    = Math.round(limitBytes / 1048576);

  const fmt = (n) => {
    const num = Number(n) || 0;
    return num >= 1 ? `${num.toFixed(1)} MB` : `${Math.round(usedBytes / 1024)} KB`;
  };

  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-header" style={{ padding: '20px 18px 18px' }}>
        <UrqenLogo size={34} showWordmark={true} wordmarkSize={17} animated={true} />
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div className="nav-section-label">Workspace</div>
        {NAV_ITEMS.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            className={`nav-item${currentTab === id ? ' active' : ''}`}
            onClick={() => setCurrentTab(id)}
            title={label}
          >
            <span className="nav-item-icon"><Icon size={16} /></span>
            {label}
            {id === 'results' && hasActiveExperiment && (
              <span style={{ marginLeft: 'auto', opacity: 0.6 }}>
                <ChevronRight size={13} />
              </span>
            )}
          </button>
        ))}

        {isAdmin && (
          <>
            <div className="nav-section-label" style={{ marginTop: 14 }}>Governance</div>
            <button
              className={`nav-item${currentTab === 'admin' ? ' active' : ''}`}
              onClick={() => setCurrentTab('admin')}
              title="Founder Admin Console"
            >
              <span className="nav-item-icon"><Shield size={16} color="var(--primary)" /></span>
              Admin Console
              <span
                style={{
                  marginLeft: 'auto',
                  fontSize: 10,
                  fontWeight: 700,
                  padding: '2px 6px',
                  borderRadius: 4,
                  background: 'rgba(0, 229, 195, 0.12)',
                  color: '#00e5c3',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                Founder
              </span>
            </button>
          </>
        )}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className="status-row">
          <span className={`status-dot ${backendOnline ? 'online' : 'offline'}`} />
          {backendOnline ? 'ML Engine Online' : 'Engine Offline'}
        </div>

        <div className="quota-track">
          <div
            className={`quota-fill${pct > 85 ? ' danger' : ''}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="quota-label">
          Storage: {fmt(usedMB)} / {limitMB} MB ({pct}%)
        </div>
      </div>
    </aside>
  );
}
