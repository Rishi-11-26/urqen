import React from 'react';
import { ShieldCheck } from 'lucide-react';
import UrqenLogo from './UrqenLogo';

export default function Footer({ onOpenLegal }) {
  return (
    <footer
      style={{
        marginTop: 48,
        padding: '22px 32px',
        borderTop: '1px solid var(--border-color)',
        background: 'var(--bg-card)',
        borderRadius: 'var(--radius-md) var(--radius-md) 0 0',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 16,
          fontSize: 12,
          color: 'var(--text-subtle)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <UrqenLogo size={22} showWordmark={false} />
          <span style={{ fontWeight: 700, color: 'var(--text-main)', letterSpacing: '0.02em' }}>
            URQEN
          </span>
          <span style={{ color: 'var(--border-color)' }}>|</span>
          <span>
            Autonomous Tabular ML Studio
          </span>
        </div>

        {/* Legal Links */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          <button
            onClick={() => onOpenLegal?.('terms')}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--text-muted)',
              fontSize: 12,
              padding: 0,
              textDecoration: 'none',
              transition: 'color var(--duration-fast)',
            }}
            onMouseEnter={(e) => (e.target.style.color = 'var(--primary)')}
            onMouseLeave={(e) => (e.target.style.color = 'var(--text-muted)')}
          >
            Terms of Service
          </button>
          <span>·</span>
          <button
            onClick={() => onOpenLegal?.('privacy')}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--text-muted)',
              fontSize: 12,
              padding: 0,
              textDecoration: 'none',
              transition: 'color var(--duration-fast)',
            }}
            onMouseEnter={(e) => (e.target.style.color = 'var(--primary)')}
            onMouseLeave={(e) => (e.target.style.color = 'var(--text-muted)')}
          >
            Privacy Policy
          </button>
          <span>·</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--success)' }}>
            <ShieldCheck size={13} />
            <span style={{ fontSize: 11.5, fontWeight: 500 }}>Zero Data-Sharing</span>
          </div>
        </div>

        <div>
          © {new Date().getFullYear()} URQEN. All rights reserved.
        </div>
      </div>
    </footer>
  );
}
