import { useState, useEffect } from 'react';
import { Database, LogOut } from 'lucide-react';

export default function Header({ title, activeDataset, mode, setMode, isBenchmarking = false, user, onSignOut }) {
  const [timeStr, setTimeStr] = useState(() =>
    new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  );

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeStr(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const dsName = activeDataset?.name || activeDataset?.filename;
  const rowCnt = activeDataset?.row_count ?? activeDataset?.n_rows ?? 0;
  const colCnt = activeDataset?.column_count ?? activeDataset?.n_columns ?? 0;

  const initials = user?.email
    ? user.email.slice(0, 2).toUpperCase()
    : '??';

  return (
    <header className="header">
      <div className="header-title">{title}</div>

      {activeDataset && (
        <div className="header-dataset-tag" title={dsName}>
          <Database size={13} />
          <span className="truncate">{dsName}</span>
          <span style={{ color: 'var(--text-subtle)', fontWeight: 400, flexShrink: 0 }}>
            · {rowCnt.toLocaleString()} × {colCnt}
          </span>
        </div>
      )}

      {/* Live System Clock */}
      <div
        className="header-clock"
        title="Live System Time"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 7,
          padding: '5px 11px',
          borderRadius: 20,
          background: 'var(--bg-subtle)',
          border: '1px solid var(--border-color)',
          fontSize: 12,
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
          letterSpacing: '0.02em',
        }}
      >
        <span
          style={{
            width: 7,
            height: 7,
            borderRadius: '50%',
            background: 'var(--success)',
            boxShadow: '0 0 6px var(--success)',
            display: 'inline-block',
          }}
        />
        <span>{timeStr}</span>
      </div>

      <div className="mode-toggle">
        <button
          className={`mode-btn${mode === 'guided' ? ' active' : ''}`}
          onClick={() => !isBenchmarking && setMode('guided')}
          disabled={isBenchmarking}
          title={isBenchmarking ? 'Configuration locked during benchmark' : 'Switch to Guided Mode'}
          style={isBenchmarking ? { cursor: 'not-allowed', opacity: 0.6 } : undefined}
        >
          Guided
        </button>
        <button
          className={`mode-btn${mode === 'expert' ? ' active' : ''}`}
          onClick={() => !isBenchmarking && setMode('expert')}
          disabled={isBenchmarking}
          title={isBenchmarking ? 'Configuration locked during benchmark' : 'Switch to Expert Mode'}
          style={isBenchmarking ? { cursor: 'not-allowed', opacity: 0.6 } : undefined}
        >
          Expert
        </button>
      </div>

      {/* User Profile Pill */}
      {user && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginLeft: 8 }}>
          <div
            title={user.email}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '5px 10px 5px 6px',
              borderRadius: 20,
              background: 'var(--bg-subtle)',
              border: '1px solid var(--border-color)',
              cursor: 'default',
            }}
          >
            <div
              style={{
                width: 26,
                height: 26,
                borderRadius: '50%',
                background: 'var(--primary)',
                color: '#080d17',
                fontSize: 11,
                fontWeight: 800,
                fontFamily: 'var(--font-mono)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}
            >
              {initials}
            </div>
            <span
              style={{
                fontSize: 12.5,
                fontWeight: 600,
                color: 'var(--text-main)',
                maxWidth: 140,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {user.email}
            </span>
          </div>
          <button
            onClick={onSignOut}
            title="Sign out"
            className="btn btn-ghost btn-sm"
            style={{ padding: '6px 8px', display: 'flex', alignItems: 'center', gap: 5 }}
          >
            <LogOut size={14} />
            <span style={{ fontSize: 12 }}>Sign Out</span>
          </button>
        </div>
      )}
    </header>
  );
}
