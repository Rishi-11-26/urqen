import React, { useState, useEffect } from 'react';
import {
  Shield,
  Users,
  Database,
  FlaskConical,
  HardDrive,
  Trash2,
  RefreshCw,
  Edit2,
  CheckCircle2,
  AlertTriangle,
  Activity,
  Server,
  Cloud,
  Clock,
  Sparkles,
  UserX,
  Download,
} from 'lucide-react';
import { api } from '../api';

export default function AdminView({ currentUserId }) {
  const [activeSubTab, setActiveSubTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [overview, setOverview] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [purging, setPurging] = useState(false);
  const [purgingAll, setPurgingAll] = useState(false);
  const [exportingHistory, setExportingHistory] = useState(false);
  const [purgeResult, setPurgeResult] = useState(null);
  const [purgeAllResult, setPurgeAllResult] = useState(null);

  // Quota edit state
  const [editingUserId, setEditingUserId] = useState(null);
  const [newQuotaMb, setNewQuotaMb] = useState(50);
  const [quotaSaving, setQuotaSaving] = useState(false);
  const [removingUserId, setRemovingUserId] = useState(null);
  const [liveUsers, setLiveUsers] = useState([]);
  const [liveUsersError, setLiveUsersError] = useState(null);

  useEffect(() => {
    loadAdminData();
  }, []);

  const loadAdminData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [ovData, expData, dsData] = await Promise.all([
        api.getAdminOverview(),
        api.getAdminExperiments(),
        api.getAdminDatasets(),
      ]);
      setOverview(ovData);
      setExperiments(expData || []);
      setDatasets(dsData || []);
    } catch (err) {
      console.error('Failed to load admin data:', err);
      setError(err.message || 'Failed to authenticate administrative access.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeSubTab !== 'users' && activeSubTab !== 'live') return undefined;

    let isMounted = true;
    const refreshLiveUsers = async () => {
      try {
        const users = await api.getAdminLiveUsers();
        if (isMounted) {
          setLiveUsers(users);
          setLiveUsersError(null);
        }
      } catch (err) {
        if (isMounted) {
          console.error('Failed to refresh live users:', err);
          setLiveUsersError(err.message || 'Could not load live users.');
        }
      }
    };

    refreshLiveUsers();
    const interval = window.setInterval(refreshLiveUsers, 15000);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, [activeSubTab]);

  const handlePurgeRetention = async () => {
    if (!window.confirm('Execute 14-day artifact cleanup? This will remove heavy .joblib model files older than 14 days to free up cloud storage, while keeping all reports and manifests.')) {
      return;
    }
    setPurging(true);
    try {
      const res = await api.purgeRetentionArtifacts(14);
      setPurgeResult(res);
      await loadAdminData();
    } catch (err) {
      alert(`Purge failed: ${err.message}`);
    } finally {
      setPurging(false);
    }
  };

  const handlePurgeAllUserData = async () => {
    const confirmation = window.prompt(
      'This permanently deletes all URQEN datasets, experiment artifacts, history, metadata and quota records for every user. Login accounts will NOT be deleted. Users will be notified the next time they open URQEN. Export the PDF first if you need an archive. Provider storage outside URQEN\'s configured local dataset/artifact directories is not included. Type PURGE ALL USER DATA to continue.'
    );
    if (confirmation !== 'PURGE ALL USER DATA') return;

    setPurgingAll(true);
    setPurgeAllResult(null);
    try {
      const result = await api.purgeAllUserData(confirmation);
      setPurgeAllResult(result);
      await loadAdminData();
      if (!result.success) {
        window.alert(`The purge completed with errors:\n${result.errors.join('\n')}`);
      }
    } catch (err) {
      window.alert(`Bulk purge failed: ${err.message}`);
    } finally {
      setPurgingAll(false);
    }
  };

  const handleExportHistory = async () => {
    setExportingHistory(true);
    try {
      const pdf = await api.downloadAdminHistoryPdf();
      const url = window.URL.createObjectURL(pdf);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'urqen-founder-history.pdf';
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => window.URL.revokeObjectURL(url), 1000);
    } catch (err) {
      window.alert(`History export failed: ${err.message}`);
    } finally {
      setExportingHistory(false);
    }
  };

  const handleUpdateQuota = async (userId) => {
    setQuotaSaving(true);
    try {
      await api.updateUserQuota(userId, Number(newQuotaMb));
      setEditingUserId(null);
      await loadAdminData();
    } catch (err) {
      alert(`Failed to update quota: ${err.message}`);
    } finally {
      setQuotaSaving(false);
    }
  };

  const handleRemoveUser = async (user) => {
    if (user.user_id === currentUserId) {
      window.alert('The active founder account cannot be removed from its own signed-in session.');
      return;
    }

    const email = user.email || 'Email unavailable';
    if (!window.confirm(
      `Permanently remove ${email}? This deletes their account, datasets, experiments, and stored artifacts. Deleted datasets will no longer appear when they sign in. This cannot be undone.`
    )) {
      return;
    }

    setRemovingUserId(user.user_id);
    try {
      const result = await api.adminDeleteUser(user.user_id);
      if (!result.details?.success) {
        await loadAdminData();
        const failures = result.details?.errors?.join(' ') || 'One or more cloud deletions failed.';
        if (result.details?.supabase_auth_deleted) {
          window.alert(`The user's Supabase login account was deleted, but some URQEN cloud records may remain: ${failures}`);
        } else if (result.details?.supabase_auth_deleted === false) {
          window.alert(`URQEN's local data was removed, but Supabase did not confirm deleting this login account. The account may still be able to sign in. ${failures}`);
        } else {
          window.alert(`URQEN data was removed. No Supabase Auth account was configured for deletion. ${failures}`);
        }
        return;
      }
      await loadAdminData();
    } catch (err) {
      alert(`User removal failed: ${err.message}`);
    } finally {
      setRemovingUserId(null);
    }
  };

  const handleDeleteExp = async (expId) => {
    if (!window.confirm(`Force delete experiment ${expId}? This will remove its artifacts.`)) return;
    try {
      await api.adminDeleteExperiment(expId);
      await loadAdminData();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    }
  };

  const handleDeleteDs = async (dsId) => {
    if (!window.confirm(`Permanently delete dataset ${dsId} from storage and remove it from its owner's account? This cannot be undone.`)) return;
    try {
      const result = await api.adminDeleteDataset(dsId);
      if (!result.success) {
        throw new Error(result.message || 'The dataset was not deleted.');
      }
      await loadAdminData();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    }
  };

  const fmtBytes = (bytes) => {
    const b = Number(bytes) || 0;
    if (b < 1024) return `${b} B`;
    if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
    if (b < 1073741824) return `${(b / 1048576).toFixed(1)} MB`;
    return `${(b / 1073741824).toFixed(2)} GB`;
  };

  if (loading && !overview) {
    return (
      <div className="view-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
        <div style={{ textAlign: 'center' }}>
          <div className="spinner" style={{ width: 36, height: 36, margin: '0 auto 16px' }} />
          <div style={{ fontSize: 14, color: 'var(--text-muted)' }}>Loading Founder Admin Console…</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="view-container" style={{ maxWidth: 700, margin: '40px auto' }}>
        <div className="alert alert-danger" style={{ padding: 24 }}>
          <AlertTriangle size={24} style={{ flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 6 }}>Administrative Access Denied</div>
            <div style={{ fontSize: 13, lineHeight: 1.5 }}>{error}</div>
            <div style={{ marginTop: 12, fontSize: 12, opacity: 0.85 }}>
              Tip: Ensure your email is included in <code>ADMIN_EMAILS</code> in your <code>.env</code> file.
            </div>
          </div>
        </div>
      </div>
    );
  }

  const globalUsed = overview?.total_storage_used_bytes || 0;
  const globalCap = overview?.global_capacity_bytes || 1073741824; // 1 GB
  const globalPct = Math.min(100, ((globalUsed / globalCap) * 100).toFixed(1));
  const usersList = overview?.users || [];
  const supabaseRowsDeleted = Object.values(purgeAllResult?.supabase_rows_deleted || {})
    .reduce((total, count) => total + (Number(count) || 0), 0);
  const liveUserIds = new Set(liveUsers.map((user) => user.user_id));
  const getUserEmail = (userId) => {
    const account = usersList.find((item) => item.user_id === userId);
    return account?.email || (userId === 'local_dev_user' ? 'Local developer workspace' : 'Email unavailable');
  };

  return (
    <div className="view-container anim-fade" style={{ maxWidth: 1200, paddingBottom: 60 }}>
      {/* Top Banner */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 24,
          padding: '20px 24px',
          background: 'linear-gradient(135deg, rgba(0,229,195,0.08), rgba(56,189,248,0.05))',
          borderRadius: 'var(--radius-lg)',
          border: '1px solid rgba(0,229,195,0.18)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div
            style={{
              width: 44,
              height: 44,
              borderRadius: 12,
              background: 'rgba(0,229,195,0.12)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--primary)',
              boxShadow: '0 4px 18px rgba(0,229,195,0.08)',
            }}
          >
            <Shield size={24} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <h2 style={{ fontSize: 20, fontWeight: 800, margin: 0, color: 'var(--text-main)' }}>
                Founder Control Console
              </h2>
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  padding: '2px 8px',
                  borderRadius: 20,
                  background: 'rgba(0,229,195,0.12)',
                  color: 'var(--primary)',
                }}
              >
                Super Admin
              </span>
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 2 }}>
              Multi-tenant user oversight, storage quotas, and 14-day retention automation
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            onClick={loadAdminData}
            className="btn btn-secondary btn-sm"
            style={{ gap: 6 }}
            disabled={loading}
          >
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
            Refresh
          </button>
          <button
            onClick={handlePurgeRetention}
            className="btn btn-primary btn-sm"
            style={{ gap: 6, background: '#0c8f7b' }}
            disabled={purging}
            title="Clean model bundle .joblib files older than 14 days"
          >
            <Clock size={14} />
            {purging ? 'Purging…' : 'Run 14d Retention Purge'}
          </button>
          <button
            onClick={handlePurgeAllUserData}
            className="btn btn-sm"
            style={{ gap: 6, background: '#991b1b', color: '#fff', border: '1px solid #b91c1c' }}
            disabled={purgingAll}
            title="Delete all users' URQEN data while retaining every login account"
          >
            <Trash2 size={14} />
            {purgingAll ? 'Purging all data…' : 'Purge All User Data'}
          </button>
          <button
            onClick={handleExportHistory}
            className="btn btn-secondary btn-sm"
            style={{ gap: 6 }}
            disabled={exportingHistory}
            title="Download account, dataset, and experiment history as a PDF"
          >
            <Download size={14} />
            {exportingHistory ? 'Preparing PDF…' : 'Export History PDF'}
          </button>
        </div>
      </div>

      {purgeAllResult && (
        <div
          className={`alert ${purgeAllResult.success ? 'alert-success' : 'alert-danger'}`}
          style={{ marginBottom: 20, padding: '12px 16px' }}
          role="status"
        >
          {purgeAllResult.success ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
          <span style={{ fontSize: 13 }}>
            <strong>{purgeAllResult.success ? 'Bulk purge complete.' : 'Bulk purge completed with errors.'}</strong>{' '}
            Cleared data for {purgeAllResult.processed_users} account(s); all {purgeAllResult.retained_auth_accounts} login account(s) were retained,
            and cleared {fmtBytes(purgeAllResult.cleared_storage_bytes)} from {purgeAllResult.cleared_storage_entries} local storage entr{purgeAllResult.cleared_storage_entries === 1 ? 'y' : 'ies'}.
            Supabase cleanup: {purgeAllResult.supabase_storage_objects_deleted} Storage object(s) ({fmtBytes(purgeAllResult.supabase_storage_bytes_deleted)}) and {supabaseRowsDeleted} metadata row(s) deleted. Login accounts remain available.
          </span>
          <button
            onClick={() => setPurgeAllResult(null)}
            style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', fontWeight: 700 }}
            aria-label="Dismiss bulk purge result"
          >
            ✕
          </button>
        </div>
      )}

      {/* Retention Purge Alert */}
      {purgeResult && (
        <div
          className="alert alert-success"
          style={{
            marginBottom: 20,
            padding: '12px 16px',
            background: '#f0fdf4',
            border: '1px solid #bbf7d0',
            color: '#166534',
          }}
        >
          <CheckCircle2 size={16} />
          <span style={{ fontSize: 13 }}>
            <strong>Retention Purge Complete:</strong> Freed {purgeResult.results?.freed_mb || 0} MB across{' '}
            {purgeResult.results?.purged_experiments_count || 0} expired model artifact(s).
          </span>
          <button
            onClick={() => setPurgeResult(null)}
            style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: '#166534', fontWeight: 700 }}
          >
            ✕
          </button>
        </div>
      )}

      {/* KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <button
          type="button"
          className="card admin-kpi-card"
          style={{ padding: 18 }}
          onClick={() => setActiveSubTab('users')}
          aria-label={`Active users: ${overview?.total_users || 0}. Open user list.`}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)' }}>ACTIVE USERS</span>
            <Users size={18} color="var(--primary)" />
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--text-main)' }}>
            {overview?.total_users || 0}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-subtle)', marginTop: 4 }}>
            Open the user directory
          </div>
        </button>

        <div className="card" style={{ padding: 18 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)' }}>HOSTED DATASETS</span>
            <Database size={18} color="#059669" />
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--text-main)' }}>
            {overview?.total_datasets || 0}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-subtle)', marginTop: 4 }}>
            {fmtBytes(overview?.total_dataset_bytes)} across users
          </div>
        </div>

        <div className="card" style={{ padding: 18 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)' }}>BENCHMARKS RUN</span>
            <FlaskConical size={18} color="#d97706" />
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--text-main)' }}>
            {overview?.total_experiments || 0}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-subtle)', marginTop: 4 }}>
            {overview?.completed_experiments || 0} completed · {overview?.failed_experiments || 0} failed
          </div>
        </div>

        <div className="card" style={{ padding: 18 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)' }}>GLOBAL CLOUD USAGE</span>
            <HardDrive size={18} color="#7c3aed" />
          </div>
          <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--text-main)' }}>
            {fmtBytes(globalUsed)}{' '}
            <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-muted)' }}>/ 1 GB</span>
          </div>
          <div style={{ marginTop: 8 }}>
            <div style={{ height: 6, background: 'var(--bg-subtle)', borderRadius: 3, overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${globalPct}%`,
                  background: globalPct > 80 ? '#ef4444' : 'var(--primary)',
                  transition: 'width 0.3s',
                }}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-subtle)', marginTop: 4 }}>
              <span>{globalPct}% consumed</span>
              <span style={{ color: '#059669', fontWeight: 600 }}>Safe Tier</span>
            </div>
          </div>
        </div>
      </div>

      {/* Supabase & Cloud Infrastructure Health Bar */}
      <div
        className="card"
        style={{
          padding: '14px 20px',
          marginBottom: 24,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'var(--bg-card)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Server size={16} color="var(--primary)" />
            <span style={{ fontSize: 13, fontWeight: 600 }}>FastAPI ML Engine:</span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12.5, color: '#059669', fontWeight: 600 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981' }} />
              Running (Port 8000)
            </span>
          </div>

          <div style={{ width: 1, height: 20, background: 'var(--border-color)' }} />

          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Cloud size={16} color="#0284c7" />
            <span style={{ fontSize: 13, fontWeight: 600 }}>Supabase Sync:</span>
            {overview?.supabase?.connected ? (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12.5, color: '#059669', fontWeight: 600 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981' }} />
                Postgres & Buckets Connected
              </span>
            ) : (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12.5, color: '#d97706', fontWeight: 600 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#f59e0b' }} />
                Local Standalone Mode
              </span>
            )}
          </div>
        </div>

        <div style={{ fontSize: 12, color: 'var(--text-subtle)' }}>
          Auto-Purge Policy: <strong style={{ color: 'var(--text-main)' }}>14-Day Artifact Retention</strong>
        </div>
      </div>

      {/* Sub Navigation Tabs */}
      <div style={{ display: 'flex', gap: 8, borderBottom: '1px solid var(--border-color)', marginBottom: 20 }}>
        {[
          { id: 'overview', label: 'Storage & Quotas', icon: HardDrive },
          { id: 'users', label: `Active Users (${usersList.length})`, icon: Users },
          { id: 'live', label: `Live Now (${liveUsers.length})`, icon: Activity },
          { id: 'experiments', label: `All Experiments (${experiments.length})`, icon: FlaskConical },
          { id: 'datasets', label: `All Datasets (${datasets.length})`, icon: Database },
          { id: 'infra', label: 'Cloud Architecture & Scaling Guide', icon: Sparkles },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeSubTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '10px 16px',
                border: 'none',
                background: 'none',
                fontSize: 13.5,
                fontWeight: isActive ? 700 : 500,
                color: isActive ? 'var(--primary)' : 'var(--text-muted)',
                borderBottom: isActive ? '2px solid var(--primary)' : '2px solid transparent',
                cursor: 'pointer',
                marginBottom: -1,
                transition: 'all 0.15s',
              }}
            >
              <Icon size={15} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* TAB 1: USERS & STORAGE QUOTAS */}
      {activeSubTab === 'overview' && (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-main)' }}>User Storage Quota Enforcement</div>
              <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
                Each user is assigned a 50 MB standard quota to guarantee fair-share usage and prevent cloud storage exhaustion.
              </div>
              <div className="admin-table-scroll-hint">Scroll horizontally to view all columns</div>
            </div>
          </div>

          <div
            className="admin-quota-table-scroll"
            role="region"
            aria-label="User storage quota table"
            tabIndex={0}
          >
          <table
            className="admin-quota-table"
            style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 13 }}
          >
            <thead>
              <tr style={{ background: 'var(--bg-subtle)', borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)', fontWeight: 600 }}>
                <th style={{ padding: '12px 20px' }}>Account Email</th>
                <th style={{ padding: '12px 16px' }}>Datasets</th>
                <th style={{ padding: '12px 16px' }}>Experiments</th>
                <th style={{ padding: '12px 16px' }}>Storage Consumed</th>
                <th style={{ padding: '12px 16px' }}>Quota Limit</th>
                <th style={{ padding: '12px 16px' }}>Capacity Used</th>
                <th style={{ padding: '12px 16px' }}>Last Active</th>
                <th style={{ padding: '12px 20px', textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {usersList.length === 0 ? (
                <tr>
                  <td colSpan={8} style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
                    No users recorded yet.
                  </td>
                </tr>
              ) : (
                usersList.map((u) => {
                  const isEditing = editingUserId === u.user_id;
                  const pct = u.quota_used_pct || 0;
                  const isAlert = pct >= 80;

                  return (
                    <tr key={u.user_id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '14px 20px' }}>
                        <div style={{ fontWeight: 600, color: 'var(--text-main)' }}>
                          {u.email || getUserEmail(u.user_id)}
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--text-subtle)', fontFamily: 'var(--font-mono)' }}>
                          {u.user_id === 'local_dev_user' ? 'Local workspace' : u.user_id}
                        </div>
                      </td>
                      <td style={{ padding: '14px 16px', fontWeight: 600 }}>{u.dataset_count}</td>
                      <td style={{ padding: '14px 16px', fontWeight: 600 }}>{u.experiment_count}</td>
                      <td style={{ padding: '14px 16px' }}>
                        <div style={{ fontWeight: 600, color: isAlert ? '#ef4444' : 'var(--text-main)' }}>
                          {fmtBytes(u.total_bytes)}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-subtle)' }}>
                          Data: {fmtBytes(u.dataset_bytes)} · Art: {fmtBytes(u.artifact_bytes)}
                        </div>
                      </td>
                      <td style={{ padding: '14px 16px' }}>
                        {isEditing ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <input
                              type="number"
                              min="10"
                              max="1000"
                              value={newQuotaMb}
                              onChange={(e) => setNewQuotaMb(e.target.value)}
                              style={{
                                width: 70,
                                padding: '4px 6px',
                                border: '1px solid var(--primary)',
                                borderRadius: 4,
                                fontSize: 12,
                              }}
                            />
                            <span style={{ fontSize: 12 }}>MB</span>
                          </div>
                        ) : (
                          <span style={{ fontWeight: 600 }}>{fmtBytes(u.quota_bytes)}</span>
                        )}
                      </td>
                      <td style={{ padding: '14px 16px', minWidth: 140 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ flex: 1, height: 6, background: 'var(--bg-subtle)', borderRadius: 3, overflow: 'hidden' }}>
                            <div
                              style={{
                                height: '100%',
                                width: `${Math.min(100, pct)}%`,
                                background: pct >= 90 ? '#ef4444' : pct >= 75 ? '#f59e0b' : '#10b981',
                              }}
                            />
                          </div>
                          <span style={{ fontSize: 11, fontWeight: 700, color: pct >= 90 ? '#ef4444' : 'var(--text-muted)' }}>
                            {pct}%
                          </span>
                        </div>
                      </td>
                      <td style={{ padding: '14px 16px', fontSize: 12, color: 'var(--text-muted)' }}>
                        {u.last_active !== 'N/A' ? new Date(u.last_active).toLocaleDateString() : '—'}
                      </td>
                      <td style={{ padding: '14px 20px', textAlign: 'right' }}>
                        {isEditing ? (
                          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6 }}>
                            <button
                              onClick={() => handleUpdateQuota(u.user_id)}
                              className="btn btn-primary btn-sm"
                              style={{ padding: '4px 10px', fontSize: 11 }}
                              disabled={quotaSaving}
                            >
                              Save
                            </button>
                            <button
                              onClick={() => setEditingUserId(null)}
                              className="btn btn-secondary btn-sm"
                              style={{ padding: '4px 10px', fontSize: 11 }}
                            >
                              Cancel
                            </button>
                          </div>
                        ) : (
                          <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 6 }}>
                            <button
                              onClick={() => {
                                setEditingUserId(u.user_id);
                                setNewQuotaMb(Math.round(u.quota_bytes / 1048576));
                              }}
                              className="btn btn-ghost btn-sm"
                              style={{ padding: '4px 8px', fontSize: 12, display: 'inline-flex', alignItems: 'center', gap: 4 }}
                            >
                              <Edit2 size={12} />
                              Adjust Quota
                            </button>
                            <button
                              onClick={() => handleRemoveUser(u)}
                              className="btn btn-ghost btn-sm"
                              style={{ padding: '4px 8px', fontSize: 12, display: 'inline-flex', alignItems: 'center', gap: 4, color: 'var(--danger-text)' }}
                              disabled={removingUserId === u.user_id}
                              title={u.user_id === currentUserId
                                ? 'The active founder account cannot be removed from this session.'
                                : `Permanently remove ${u.email || u.user_id}`}
                            >
                              <UserX size={13} />
                              {removingUserId === u.user_id ? 'Removing…' : 'Remove'}
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
          </div>
        </div>
      )}

      {activeSubTab === 'users' && (
        <div className="card admin-users-panel">
          <div className="card-header">
            <div>
              <div className="card-title">User Directory</div>
              <div className="card-subtitle">Registered accounts, recent activity, and storage footprint.</div>
            </div>
            <span className="admin-presence-note">
              {liveUsersError ? 'Presence unavailable' : `${liveUsers.length} live now`}
            </span>
          </div>
          <div className="table-container">
            <table className="data-table admin-users-table">
              <thead>
                <tr>
                  <th>Account</th>
                  <th>Presence</th>
                  <th>Datasets</th>
                  <th>Experiments</th>
                  <th>Storage</th>
                  <th>Last Activity</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {usersList.length === 0 ? (
                  <tr><td colSpan={7} className="admin-empty-cell">No user accounts found.</td></tr>
                ) : usersList.map((account) => (
                  <tr key={account.user_id}>
                    <td>
                      <strong>{account.email || getUserEmail(account.user_id)}</strong>
                      <small>{account.user_id}</small>
                    </td>
                    <td>
                      <span className={`admin-presence-status${liveUserIds.has(account.user_id) ? ' is-live' : ''}`}>
                        <span />
                        {liveUsersError ? 'Unknown' : liveUserIds.has(account.user_id) ? 'Live now' : 'Offline'}
                      </span>
                    </td>
                    <td>{account.dataset_count}</td>
                    <td>{account.experiment_count}</td>
                    <td>{fmtBytes(account.total_bytes)}</td>
                    <td>{account.last_active !== 'N/A' ? new Date(account.last_active).toLocaleString() : 'No activity recorded'}</td>
                    <td>
                      <button
                        onClick={() => handleRemoveUser(account)}
                        className="btn btn-ghost btn-sm admin-remove-user"
                        disabled={removingUserId === account.user_id}
                        title={account.user_id === currentUserId
                          ? 'The active founder account cannot be removed from this session.'
                          : `Permanently remove ${account.email || account.user_id}`}
                      >
                        <UserX size={13} />
                        {removingUserId === account.user_id ? 'Removing…' : 'Remove'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeSubTab === 'live' && (
        <div className="card admin-live-panel">
          <div className="card-header">
            <div>
              <div className="card-title">Live Users</div>
              <div className="card-subtitle">A user is considered live while their signed-in workspace sends a foreground heartbeat.</div>
            </div>
            <span className="admin-live-count"><span />{liveUsers.length} online</span>
          </div>
          {liveUsersError ? (
            <div className="admin-live-message is-error">{liveUsersError}</div>
          ) : liveUsers.length === 0 ? (
            <div className="admin-live-message">No signed-in workspaces have sent a recent heartbeat.</div>
          ) : (
            <div className="table-container">
              <table className="data-table admin-live-table">
                <thead>
                  <tr><th>Account</th><th>Presence</th><th>Last heartbeat</th></tr>
                </thead>
                <tbody>
                  {liveUsers.map((user) => (
                    <tr key={user.user_id}>
                      <td>
                        <strong>{user.email || 'Email unavailable'}</strong>
                        <small>{user.user_id}</small>
                      </td>
                      <td><span className="admin-presence-status is-live"><span />Live now</span></td>
                      <td>{new Date(user.last_seen).toLocaleTimeString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="admin-presence-footnote">
            Presence expires after 90 seconds without a heartbeat and refreshes every 15 seconds. Data is held in this backend process and resets on restart.
          </div>
        </div>
      )}

      {/* TAB 2: ALL EXPERIMENTS */}
      {activeSubTab === 'experiments' && (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-main)' }}>Global Experiments Inventory</div>
            <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
              All benchmarks evaluated across the platform with winning algorithms and tiebreaking audits.
            </div>
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 13 }}>
            <thead>
              <tr style={{ background: 'var(--bg-subtle)', borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)', fontWeight: 600 }}>
                <th style={{ padding: '12px 20px' }}>Experiment ID</th>
                <th style={{ padding: '12px 16px' }}>User</th>
                <th style={{ padding: '12px 16px' }}>Dataset</th>
                <th style={{ padding: '12px 16px' }}>Task</th>
                <th style={{ padding: '12px 16px' }}>Winning Model</th>
                <th style={{ padding: '12px 16px' }}>Score</th>
                <th style={{ padding: '12px 16px' }}>Status</th>
                <th style={{ padding: '12px 16px' }}>Created</th>
                <th style={{ padding: '12px 20px', textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {experiments.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
                    No experiments found.
                  </td>
                </tr>
              ) : (
                experiments.map((exp) => (
                  <tr key={exp.experiment_id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '12px 20px', fontFamily: 'monospace', fontSize: 12 }}>
                      {exp.experiment_id.slice(0, 8)}…
                    </td>
                    <td style={{ padding: '12px 16px', fontSize: 12, color: 'var(--text-muted)' }}>
                      {getUserEmail(exp.user_id)}
                    </td>
                    <td style={{ padding: '12px 16px', fontWeight: 600 }}>{exp.dataset_name || '—'}</td>
                    <td style={{ padding: '12px 16px' }}>
                      <span className="badge badge-secondary" style={{ fontSize: 11 }}>
                        {exp.task_type}
                      </span>
                    </td>
                    <td style={{ padding: '12px 16px', fontWeight: 600, color: 'var(--primary)' }}>
                      {exp.best_model || '—'}
                    </td>
                    <td style={{ padding: '12px 16px', fontWeight: 700 }}>
                      {exp.results?.models?.[0]?.holdout_score != null
                        ? Number(exp.results.models[0].holdout_score).toFixed(4)
                        : '—'}
                    </td>
                    <td style={{ padding: '12px 16px' }}>
                      <span
                        style={{
                          fontSize: 11,
                          fontWeight: 700,
                          padding: '2px 8px',
                          borderRadius: 12,
                          background: exp.status === 'completed' ? '#dcfce7' : exp.status === 'failed' ? '#fee2e2' : '#fef3c7',
                          color: exp.status === 'completed' ? '#166534' : exp.status === 'failed' ? '#991b1b' : '#92400e',
                        }}
                      >
                        {exp.status}
                      </span>
                    </td>
                    <td style={{ padding: '12px 16px', fontSize: 12, color: 'var(--text-subtle)' }}>
                      {exp.created_at ? new Date(exp.created_at).toLocaleDateString() : '—'}
                    </td>
                    <td style={{ padding: '12px 20px', textAlign: 'right' }}>
                      <button
                        onClick={() => handleDeleteExp(exp.experiment_id)}
                        className="btn btn-ghost btn-sm"
                        style={{ color: '#ef4444', padding: '4px 8px' }}
                        title="Force delete experiment and its artifacts"
                      >
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* TAB 3: ALL DATASETS */}
      {activeSubTab === 'datasets' && (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-main)' }}>Global Datasets Inventory</div>
            <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
              All CSV and Excel files uploaded to the platform across all accounts.
            </div>
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 13 }}>
            <thead>
              <tr style={{ background: 'var(--bg-subtle)', borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)', fontWeight: 600 }}>
                <th style={{ padding: '12px 20px' }}>Dataset Name</th>
                <th style={{ padding: '12px 16px' }}>Owner</th>
                <th style={{ padding: '12px 16px' }}>Rows × Columns</th>
                <th style={{ padding: '12px 16px' }}>Format</th>
                <th style={{ padding: '12px 16px' }}>Size</th>
                <th style={{ padding: '12px 16px' }}>Upload Date</th>
                <th style={{ padding: '12px 20px', textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {datasets.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
                    No datasets uploaded yet.
                  </td>
                </tr>
              ) : (
                datasets.map((ds) => (
                  <tr key={ds.dataset_id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '12px 20px', fontWeight: 600, color: 'var(--text-main)' }}>
                      {ds.name || ds.filename}
                    </td>
                    <td style={{ padding: '12px 16px', fontSize: 12, color: 'var(--text-muted)' }}>
                      {getUserEmail(ds.user_id)}
                    </td>
                    <td style={{ padding: '12px 16px' }}>
                      {(ds.row_count || 0).toLocaleString()} × {ds.column_count || 0}
                    </td>
                    <td style={{ padding: '12px 16px' }}>
                      <span className="badge badge-secondary" style={{ textTransform: 'uppercase', fontSize: 10 }}>
                        {ds.file_type || 'CSV'}
                      </span>
                    </td>
                    <td style={{ padding: '12px 16px' }}>{fmtBytes(ds.size_bytes)}</td>
                    <td style={{ padding: '12px 16px', fontSize: 12, color: 'var(--text-subtle)' }}>
                      {ds.created_at ? new Date(ds.created_at).toLocaleDateString() : '—'}
                    </td>
                    <td style={{ padding: '12px 20px', textAlign: 'right' }}>
                      <button
                        onClick={() => handleDeleteDs(ds.dataset_id)}
                        className="btn btn-ghost btn-sm"
                        style={{ color: '#ef4444', padding: '4px 8px' }}
                        title="Force delete dataset"
                      >
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* TAB 4: CLOUD ARCHITECTURE & SCALING GUIDE */}
      {activeSubTab === 'infra' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 18 }}>
          <div className="card" style={{ padding: 22 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <div style={{ padding: 8, borderRadius: 8, background: '#eff6ff', color: 'var(--primary)' }}>
                <Server size={20} />
              </div>
              <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>Will My Device Crash with Many Users?</h3>
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--text-muted)' }}>
              <p style={{ marginTop: 0 }}>
                URQEN runs on the infrastructure configured for this deployment, such as a cloud host or a local server.
              </p>
              <ul style={{ paddingLeft: 20, margin: '10px 0' }}>
                <li>
                  <strong>Founder Device Load:</strong> Model workloads run on the configured backend, not in the browser displaying this console.
                </li>
                <li>
                  <strong>Server Process Isolation:</strong> Benchmark capacity depends on the backend worker configuration and available compute.
                </li>
                <li>
                  <strong>Crash Protection:</strong> If a user uploads an invalid file, the error is isolated to that specific background task. The main server never crashes.
                </li>
              </ul>
            </div>
          </div>

          <div className="card" style={{ padding: 22 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <div style={{ padding: 8, borderRadius: 8, background: '#f0fdf4', color: '#16a34a' }}>
                <HardDrive size={20} />
              </div>
              <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>How 50 MB Quota Scales to 100+ Users</h3>
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--text-muted)' }}>
              <p style={{ marginTop: 0 }}>
                Storage capacity and user limits depend on the configured hosting and Supabase plans.
              </p>
              <ul style={{ paddingLeft: 20, margin: '10px 0' }}>
                <li>
                  <strong>Average Tabular Dataset:</strong> A 100,000-row CSV is only ~8 MB. 50 MB is enough for 5 to 10 typical datasets per user!
                </li>
                <li>
                  <strong>14-Day Automated Retention:</strong> Heavy <code>.joblib</code> model files expire automatically after 14 days, saving ~90% of disk space while keeping evaluation reports and charts.
                </li>
                <li>
                  <strong>Founder Quota Control:</strong> In the table on Tab 1, you can upgrade specific VIP users or researchers to 100 MB or 250 MB with 1 click.
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
