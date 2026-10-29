import React, { useState } from 'react';
import { X, Code2, Copy, Check, Download, Lock, Sparkles } from 'lucide-react';
import { api } from '../api';

export default function CodeExportModal({ isOpen, onClose, codeData, onUpgrade }) {
  const [copied, setCopied] = useState(false);

  if (!isOpen || !codeData) return null;

  const isLocked = codeData.is_locked === true;
  const displayCode = isLocked ? codeData.preview_code : codeData.code;

  const handleCopy = () => {
    if (displayCode) {
      navigator.clipboard.writeText(displayCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownloadFile = () => {
    if (isLocked) {
      onUpgrade?.();
      return;
    }
    const blob = new Blob([displayCode], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = codeData.filename || 'urqen_model_pipeline.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
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
          maxWidth: 760,
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
                background: 'linear-gradient(135deg, #eab308 0%, #f97316 100%)',
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Code2 size={17} />
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-main)' }}>
                Python Pipeline Script ({codeData.best_model || 'Winning Model'})
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                Standalone, reproducible scikit-learn code ready to paste into Google Colab or local scripts
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

        {/* Code Content */}
        <div style={{ padding: '20px 24px', flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
          {isLocked && (
            <div
              style={{
                padding: '12px 16px',
                background: 'rgba(234, 179, 8, 0.1)',
                border: '1px solid rgba(234, 179, 8, 0.3)',
                borderRadius: 'var(--radius-sm)',
                marginBottom: 14,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <Lock size={18} color="#eab308" />
                <div style={{ fontSize: 12.5, color: 'var(--text-main)', lineHeight: 1.4 }}>
                  <strong>Pro Code Export Preview:</strong> Showing pipeline structure. Unlock full executable code with a ₹99 Top-Up or Student plan.
                </div>
              </div>
              <button
                className="btn btn-primary btn-sm"
                onClick={onUpgrade}
                style={{ flexShrink: 0, fontSize: 12, padding: '6px 12px', display: 'flex', alignItems: 'center', gap: 5 }}
              >
                <Sparkles size={13} /> Unlock Full Code
              </button>
            </div>
          )}

          <div style={{ position: 'relative', flex: 1 }}>
            <pre
              style={{
                background: '#0a0f1d',
                color: '#e2e8f0',
                padding: 18,
                borderRadius: 'var(--radius-md)',
                fontFamily: 'var(--font-mono)',
                fontSize: 12,
                lineHeight: 1.6,
                overflowX: 'auto',
                margin: 0,
                border: '1px solid rgba(255,255,255,0.08)',
                maxHeight: 400,
              }}
            >
              <code>{displayCode}</code>
            </pre>
            {isLocked && (
              <div
                style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  height: 120,
                  background: 'linear-gradient(to bottom, rgba(10,15,29,0.3) 0%, rgba(10,15,29,0.98) 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '0 0 var(--radius-md) var(--radius-md)',
                }}
              >
                <button
                  className="btn btn-primary"
                  onClick={onUpgrade}
                  style={{ gap: 8, fontSize: 13 }}
                >
                  <Lock size={14} /> Unlock Complete Python Pipeline (₹99)
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div
          style={{
            padding: '14px 24px',
            borderTop: '1px solid var(--border-color)',
            background: 'var(--bg-subtle)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
            {isLocked ? 'Previewing 16 lines' : 'Full pipeline ready to run'}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className="btn btn-secondary btn-sm"
              onClick={handleCopy}
              style={{ gap: 6, fontSize: 12.5 }}
            >
              {copied ? <Check size={14} color="var(--success)" /> : <Copy size={14} />}
              {copied ? 'Copied to Clipboard' : 'Copy Code'}
            </button>
            <button
              className="btn btn-primary btn-sm"
              onClick={handleDownloadFile}
              style={{ gap: 6, fontSize: 12.5 }}
            >
              {isLocked ? <Lock size={14} /> : <Download size={14} />}
              {isLocked ? 'Download .py (Pro)' : 'Download .py File'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
