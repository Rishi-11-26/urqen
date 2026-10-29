import React from 'react';
import { X, Sparkles, ShieldCheck, Mail, CheckCircle2 } from 'lucide-react';

export default function ProUpgradeModal({ isOpen, onClose, featureTitle, featureDesc }) {
  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: 'rgba(11, 17, 32, 0.75)',
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
          maxWidth: 580,
          background: 'var(--bg-card)',
          borderRadius: 'var(--radius-lg)',
          boxShadow: 'var(--shadow-xl)',
          border: '1px solid var(--border-color)',
          overflow: 'hidden',
          animation: 'slideUp 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '18px 24px',
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'linear-gradient(135deg, rgba(37,99,235,0.1) 0%, rgba(168,85,247,0.1) 100%)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 34,
                height: 34,
                borderRadius: 'var(--radius-sm)',
                background: 'linear-gradient(135deg, #f59e0b 0%, #ec4899 100%)',
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 2px 8px rgba(245,158,11,0.3)',
              }}
            >
              <Sparkles size={18} />
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-main)' }}>
                {featureTitle || 'Upgrade to URQEN Pro'}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Unlock high-value exports, predictions, and priority compute
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

        {/* Body */}
        <div style={{ padding: '24px' }}>
          {featureDesc && (
            <div
              style={{
                padding: '12px 16px',
                background: 'rgba(245,158,11,0.08)',
                border: '1px solid rgba(245,158,11,0.25)',
                borderRadius: 'var(--radius-sm)',
                marginBottom: 20,
                fontSize: 13,
                color: 'var(--text-main)',
                lineHeight: 1.5,
              }}
            >
              {featureDesc}
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 14, marginBottom: 22 }}>
            {/* ₹99 Top-Up */}
            <div
              style={{
                padding: '16px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-subtle)',
                border: '1.5px solid rgba(245,158,11,0.4)',
                position: 'relative',
              }}
            >
              <div style={{ display: 'inline-block', fontSize: 10.5, fontWeight: 700, background: '#f59e0b', color: '#000', padding: '2px 8px', borderRadius: 10, textTransform: 'uppercase', marginBottom: 8 }}>
                Most Popular
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-main)' }}>
                ₹99 <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)' }}>one-time</span>
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#f59e0b', marginTop: 2, marginBottom: 10 }}>
                Deadline Top-Up Pack
              </div>
              <ul style={{ fontSize: 12, color: 'var(--text-muted)', paddingLeft: 18, margin: 0, lineHeight: 1.8 }}>
                <li><strong>10 Extra Experiment Runs</strong></li>
                <li>Model Bundle (.joblib) Download</li>
                <li>Python Code Export for project report</li>
                <li>No recurring subscription</li>
              </ul>
            </div>

            {/* ₹249 Student Plan */}
            <div
              style={{
                padding: '16px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-subtle)',
                border: '1.5px solid rgba(37,99,235,0.4)',
              }}
            >
              <div style={{ display: 'inline-block', fontSize: 10.5, fontWeight: 700, background: 'var(--primary)', color: '#fff', padding: '2px 8px', borderRadius: 10, textTransform: 'uppercase', marginBottom: 8 }}>
                Full Semester
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-main)' }}>
                ₹249 <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)' }}>/ month</span>
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--primary)', marginTop: 2, marginBottom: 10 }}>
                Student Pro Plan
              </div>
              <ul style={{ fontSize: 12, color: 'var(--text-muted)', paddingLeft: 18, margin: 0, lineHeight: 1.8 }}>
                <li><strong>30 Experiments / month</strong></li>
                <li>Live Predictions on new CSVs</li>
                <li>90-Day Model Artifact Retention</li>
                <li>250 MB Workspace Storage</li>
              </ul>
            </div>
          </div>

          {/* Activation Instructions */}
          <div
            style={{
              padding: '14px 18px',
              background: 'var(--bg-subtle)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 14,
              flexWrap: 'wrap',
            }}
          >
            <div>
              <div style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Mail size={14} color="var(--primary)" /> Instant UPI Activation
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 2 }}>
                Email your registered account to <strong style={{ color: 'var(--text-main)' }}>urqen0@gmail.com</strong>
              </div>
            </div>
            <a
              href="mailto:urqen0@gmail.com?subject=URQEN%20Plan%20Upgrade%20Request&body=Hi%2C%20I%20would%20like%20to%20activate%20a%20plan%20for%20my%20account.%0A%0ARegistered%20Email%3A%20%0APlan%3A%20%E2%82%B999%20Top-Up%20%2F%20%E2%82%B9249%20Student%0APayment%20Method%3A%20UPI"
              className="btn btn-primary btn-sm"
              style={{ padding: '8px 16px', fontSize: 12.5 }}
            >
              Contact Developer
            </a>
          </div>
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '12px 24px',
            borderTop: '1px solid var(--border-color)',
            background: 'var(--bg-subtle)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span style={{ fontSize: 11, color: 'var(--text-subtle)', display: 'flex', alignItems: 'center', gap: 4 }}>
            <ShieldCheck size={13} color="var(--success)" /> Instant account upgrade upon verification
          </span>
          <button className="btn btn-secondary btn-sm" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
