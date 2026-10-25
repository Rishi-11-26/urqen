import { Home, AlertCircle } from 'lucide-react';
import UrqenLogo from '../components/UrqenLogo';

export default function NotFoundView({ onNavigate }) {
  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-app)',
        padding: 24,
        textAlign: 'center',
      }}
    >
      <div style={{ marginBottom: 28 }}>
        <UrqenLogo size={44} showWordmark={true} wordmarkSize={20} />
      </div>

      <div
        style={{
          width: 72,
          height: 72,
          borderRadius: '50%',
          background: 'rgba(239,68,68,0.1)',
          border: '1px solid rgba(239,68,68,0.25)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: 24,
        }}
      >
        <AlertCircle size={32} color="#ef4444" />
      </div>

      <div
        style={{
          fontFamily: "'Syne', var(--font-sans)",
          fontSize: 'clamp(52px, 10vw, 96px)',
          fontWeight: 800,
          color: 'var(--primary)',
          letterSpacing: '-0.04em',
          lineHeight: 1,
          marginBottom: 12,
          opacity: 0.18,
        }}
      >
        404
      </div>

      <h1
        style={{
          fontFamily: "'Syne', var(--font-sans)",
          fontSize: 22,
          fontWeight: 700,
          color: 'var(--text-main)',
          marginBottom: 10,
        }}
      >
        Page not found
      </h1>

      <p style={{ fontSize: 14, color: 'var(--text-muted)', maxWidth: 380, lineHeight: 1.65, marginBottom: 28 }}>
        The URL you entered doesn't exist in URQEN. It may have been moved, deleted, or you may have mistyped it.
      </p>

      <button
        onClick={() => onNavigate?.('dashboard')}
        className="btn btn-primary"
        style={{ gap: 8, fontSize: 14, padding: '11px 24px' }}
      >
        <Home size={16} />
        Go to Dashboard
      </button>
    </div>
  );
}
