import React, { useState } from 'react';
import { X, ShieldCheck, FileText, Lock, CheckCircle, Database, AlertTriangle, UserCheck } from 'lucide-react';

export default function LegalModal({ isOpen, onClose, initialTab = 'terms' }) {
  const [activeTab, setActiveTab] = useState(initialTab);

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: 'rgba(11, 17, 32, 0.75)',
        backdropFilter: 'blur(10px)',
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
          maxWidth: 720,
          maxHeight: '88vh',
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
        {/* Modal Header */}
        <div
          style={{
            padding: '18px 24px',
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
                background: 'linear-gradient(135deg, #2563eb 0%, #06b6d4 100%)',
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 2px 8px rgba(37,99,235,0.3)',
              }}
            >
              <ShieldCheck size={18} />
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-main)' }}>
                URQEN Platform Governance & Trust Center
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                Commercial-Grade Data Ownership, Privacy Guarantees & Terms
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

        {/* Tab Navigation */}
        <div
          style={{
            display: 'flex',
            borderBottom: '1px solid var(--border-color)',
            background: 'var(--bg-card)',
            padding: '0 24px',
          }}
        >
          <button
            onClick={() => setActiveTab('terms')}
            style={{
              padding: '12px 18px',
              border: 'none',
              background: 'none',
              fontSize: 13,
              fontWeight: activeTab === 'terms' ? 700 : 500,
              color: activeTab === 'terms' ? 'var(--primary)' : 'var(--text-muted)',
              borderBottom: activeTab === 'terms' ? '2.5px solid var(--primary)' : '2.5px solid transparent',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 7,
            }}
          >
            <FileText size={15} />
            Terms of Service
          </button>
          <button
            onClick={() => setActiveTab('privacy')}
            style={{
              padding: '12px 18px',
              border: 'none',
              background: 'none',
              fontSize: 13,
              fontWeight: activeTab === 'privacy' ? 700 : 500,
              color: activeTab === 'privacy' ? 'var(--primary)' : 'var(--text-muted)',
              borderBottom: activeTab === 'privacy' ? '2.5px solid var(--primary)' : '2.5px solid transparent',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 7,
            }}
          >
            <Lock size={15} />
            Privacy Policy
          </button>
        </div>

        {/* Modal Body Content */}
        <div
          style={{
            padding: '24px 28px',
            overflowY: 'auto',
            fontSize: 13,
            lineHeight: 1.65,
            color: 'var(--text-main)',
          }}
        >
          {activeTab === 'terms' ? (
            <div>
              {/* Highlight Banner */}
              <div
                style={{
                  padding: '12px 16px',
                  background: 'var(--primary-light)',
                  border: '1px solid var(--primary-mid)',
                  borderRadius: 'var(--radius-sm)',
                  marginBottom: 18,
                  fontSize: 12.5,
                  color: 'var(--text-main)',
                }}
              >
                <strong>Customer Intellectual Property Guarantee:</strong> You retain 100% exclusive ownership of all uploaded datasets, features, prediction targets, and trained ML artifacts (.joblib model bundles, manifests, and preprocessed datasets).
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--primary)', fontWeight: 700 }}>
                <CheckCircle size={15} /> 1. User Data Ownership & Artifact Rights
              </div>
              <p style={{ color: 'var(--text-muted)', marginBottom: 16 }}>
                URQEN provides empirical machine learning workbench capabilities. All uploaded tabular files (CSV, TSV, XLSX), extracted feature sets, trained models, diagnostic charts, and preprocessed CSVs remain your exclusive property. URQEN claims no copyright, patent, trade secret, or intellectual property rights over any model trained on your data.
              </p>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--primary)', fontWeight: 700 }}>
                <AlertTriangle size={15} color="var(--warning)" /> 2. Statistical Outputs & Professional Validation
              </div>
              <p style={{ color: 'var(--text-muted)', marginBottom: 16 }}>
                All model benchmarks, evaluation metrics (e.g. Macro F1, $R^2$, ROC-AUC), cross-validation estimations, and feature importances are statistical outputs derived empirically from your supplied data sample. URQEN does not provide warranties that models are error-free or that historical cross-validation performance will replicate on out-of-distribution real-world data. Users are solely responsible for obtaining appropriate clinical, scientific, regulatory, or financial professional validation before deploying models for high-risk autonomous decision-making.
              </p>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--primary)', fontWeight: 700 }}>
                <CheckCircle size={15} /> 3. Prohibited Uses & Security Safeguards
              </div>
              <p style={{ color: 'var(--text-muted)', marginBottom: 16 }}>
                Users agree not to upload malicious scripts, corrupted executable payloads, unlawfully harvested personal identifiable information (PII), or confidential data in violation of third-party non-disclosure agreements. URQEN maintains rate limits and dataset size caps (up to 50 MB / 100,000 rows on standard tier) to protect system stability for all tenants.
              </p>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--primary)', fontWeight: 700 }}>
                <CheckCircle size={15} /> 4. Cascading Erasure & Right to Purge
              </div>
              <p style={{ color: 'var(--text-muted)', marginBottom: 16 }}>
                Users may delete individual datasets, individual experiments, or all historical execution records at any time. When deletion is triggered, URQEN executes cascading cleanup removing physical files from disk partitions and database metadata records.
              </p>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--primary)', fontWeight: 700 }}>
                <CheckCircle size={15} /> 5. Service Availability & Modifications
              </div>
              <p style={{ color: 'var(--text-muted)', marginBottom: 0 }}>
                URQEN operates under continuous refinement. Feature enhancements, additional algorithms, or maintenance updates may be deployed. Commercial subscriptions may introduce higher storage and compute allowances while preserving free tier functionality.
              </p>
            </div>
          ) : (
            <div>
              {/* Privacy Policy */}
              <div
                style={{
                  padding: '14px 16px',
                  background: 'var(--success-bg)',
                  border: '1px solid var(--success-border)',
                  borderRadius: 'var(--radius-sm)',
                  marginBottom: 18,
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 10,
                }}
              >
                <Database size={18} color="var(--success)" style={{ flexShrink: 0, marginTop: 2 }} />
                <div style={{ fontSize: 12.5, color: 'var(--success-text)', lineHeight: 1.5 }}>
                  <strong>Zero-Foundation-Model Training Pledge:</strong> We strictly do <em>not</em> use your uploaded tabular datasets, target variables, or generated model outputs to train, fine-tune, evaluate, or improve any global foundation models, third-party LLMs, or shared algorithms.
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--success)', fontWeight: 700 }}>
                <UserCheck size={15} /> 1. Data Collection & Account Scope
              </div>
              <p style={{ color: 'var(--text-muted)', marginBottom: 16 }}>
                We collect only your registered email address and authentication credentials securely managed through Supabase Auth. We store the tabular datasets and configuration overrides that you intentionally upload. We do not inspect, monetize, or broker dataset contents to advertisers or third parties.
              </p>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--success)', fontWeight: 700 }}>
                <Lock size={15} /> 2. Multi-Tenant Isolation & Storage Architecture
              </div>
              <p style={{ color: 'var(--text-muted)', marginBottom: 16 }}>
                Customer records are associated with their account user identifiers. Database records may be protected by PostgreSQL Row Level Security (RLS) policies enforcing <code>auth.uid() = user_id</code>, depending on deployment configuration. Uploaded files and generated artifacts are stored by the configured backend.
              </p>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--success)', fontWeight: 700 }}>
                <CheckCircle size={15} /> 3. Third-Party Infrastructure & Zero External AI APIs
              </div>
              <p style={{ color: 'var(--text-muted)', marginBottom: 16 }}>
                All tabular machine learning (scikit-learn pipelines, XGBoost, cross-validation, feature importance) executes locally on URQEN's dedicated Python runtime. Your raw data is <strong>never</strong> transmitted to OpenAI, Anthropic, Google, or any external AI API. Cloud persistence is hosted on Supabase (PostgreSQL with encrypted transport and storage).
              </p>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--success)', fontWeight: 700 }}>
                <CheckCircle size={15} /> 4. Cookies & Session Storage
              </div>
              <p style={{ color: 'var(--text-muted)', marginBottom: 16 }}>
                URQEN uses solely functional, essential authentication tokens stored securely in browser <code>localStorage</code> to maintain your signed-in state. We do not use third-party tracking scripts, advertising beacons, or surveillance cookies.
              </p>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: 'var(--success)', fontWeight: 700 }}>
                <CheckCircle size={15} /> 5. Data Subject Rights (GDPR / CCPA / DPDP)
              </div>
              <p style={{ color: 'var(--text-muted)', marginBottom: 0 }}>
                You have the full right to view, export, or permanently erase all stored records. Initiating dataset or experiment deletion removes files from server disks immediately. For complete account deletion or data subject inquiries, contact your service administrator.
              </p>
            </div>
          )}
        </div>

        {/* Modal Footer */}
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
          <span style={{ fontSize: 11.5, color: 'var(--text-subtle)' }}>
            Effective: {new Date().getFullYear()} · URQEN
          </span>
          <button className="btn btn-primary btn-sm" onClick={onClose}>
            Acknowledge &amp; Close
          </button>
        </div>
      </div>
    </div>
  );
}
