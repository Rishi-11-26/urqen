import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import DashboardView from './views/DashboardView';
import DatasetLabView from './views/DatasetLabView';
import ExperimentStudioView from './views/ExperimentStudioView';
import ResultsView from './views/ResultsView';
import HistoryView from './views/HistoryView';
import HowItWorksView from './views/HowItWorksView';
import AuthView from './views/AuthView';
import AdminView from './views/AdminView';
import NotFoundView from './views/NotFoundView';
import LegalModal from './components/LegalModal';
import Footer from './components/Footer';
import UrqenLogo from './components/UrqenLogo';
import { api, setTokenProvider } from './api';
import { supabase } from './lib/supabase';

function UrqenApp() {
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [mode, setMode] = useState('guided');

  // Auth state
  const [session, setSession] = useState(null);   // supabase session
  const [user, setUser] = useState(null);          // supabase user
  const [authLoading, setAuthLoading] = useState(true);
  const [launchMinimumElapsed, setLaunchMinimumElapsed] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);

  const [backendOnline, setBackendOnline] = useState(false);
  const [storageQuota, setStorageQuota] = useState(null);
  const [systemNotice, setSystemNotice] = useState(null);

  useEffect(() => {
    const timer = window.setTimeout(() => setLaunchMinimumElapsed(true), 1500);
    return () => window.clearTimeout(timer);
  }, []);

  const [datasets, setDatasets] = useState([]);
  const [activeDataset, setActiveDataset] = useState(null);
  const [targetColumn, setTargetColumn] = useState(null);
  const [taskType, setTaskType] = useState(null);

  const [experiments, setExperiments] = useState([]);
  const [activeExperiment, setActiveExperiment] = useState(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

  // Legal & Privacy modal
  const [legalModalOpen, setLegalModalOpen] = useState(false);
  const [legalModalTab, setLegalModalTab] = useState('terms');

  // Maintenance mode
  const [maintenance, setMaintenance] = useState({ active: false, message: '' });

  const handleOpenLegal = (tab = 'terms') => {
    setLegalModalTab(tab);
    setLegalModalOpen(true);
  };

  // ─────────────────── Auth bootstrap ───────────────────
  useEffect(() => {
    // Always wire the token provider — even before session is known
    setTokenProvider(() => session?.access_token ?? null);
  }, [session]);

  useEffect(() => {
    // Check for existing session on mount
    supabase.auth.getSession().then(({ data }) => {
      const s = data?.session ?? null;
      setSession(s);
      setUser(s?.user ?? null);
      setAuthLoading(false);
    });

    // Listen for sign-in / sign-out events
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, s) => {
      setSession(s);
      setUser(s?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  // Bootstrap app data whenever user is confirmed signed-in
  useEffect(() => {
    if (!authLoading) {
      bootstrapApp();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authLoading, user]);

  useEffect(() => {
    if (!session?.access_token) return undefined;

    const heartbeat = () => {
      if (document.visibilityState !== 'visible') return;
      api.sendPresenceHeartbeat().catch((err) => {
        console.warn('Presence heartbeat failed:', err);
      });
    };

    heartbeat();
    const timer = window.setInterval(heartbeat, 30000);
    document.addEventListener('visibilitychange', heartbeat);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', heartbeat);
    };
  }, [session?.access_token]);

  useEffect(() => {
    if (!session?.access_token) return undefined;

    let isMounted = true;
    const checkSystemNotice = async () => {
      try {
        const notice = await api.getAccountNotice();
        if (!isMounted || !notice?.notice_id) return;
        let dismissedNoticeId = null;
        try {
          dismissedNoticeId = window.localStorage.getItem('urqen:last-dismissed-system-notice');
        } catch (err) {
          console.warn('Could not read saved system notice state:', err);
        }
        setSystemNotice(notice.notice_id === dismissedNoticeId ? null : notice);
      } catch (err) {
        console.warn('System notice check failed:', err);
      }
    };

    checkSystemNotice();
    const timer = window.setInterval(checkSystemNotice, 30000);
    return () => {
      isMounted = false;
      window.clearInterval(timer);
    };
  }, [session?.access_token]);

  // Periodic check for platform maintenance mode
  useEffect(() => {
    let isMounted = true;
    const checkMaintenance = async () => {
      try {
        const m = await api.getMaintenanceStatus();
        if (isMounted && m) {
          setMaintenance(m);
        }
      } catch (err) {
        // silent fallback
      }
    };
    checkMaintenance();
    const interval = window.setInterval(checkMaintenance, 20000);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  // ─────────────────── Auth callbacks ───────────────────
  const handleAuthSuccess = (newSession, newUser) => {
    setSession(newSession);
    setUser(newUser);
    setTokenProvider(() => newSession?.access_token ?? null);
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    setSession(null);
    setUser(null);
    // Reset app state
    setDatasets([]);
    setActiveDataset(null);
    setExperiments([]);
    setActiveExperiment(null);
    setStorageQuota(null);
    setIsAdmin(false);
    setSystemNotice(null);
    setCurrentTab('dashboard');
  };

  const dismissSystemNotice = () => {
    if (systemNotice?.notice_id) {
      try {
        window.localStorage.setItem('urqen:last-dismissed-system-notice', systemNotice.notice_id);
      } catch (err) {
        console.warn('Could not save system notice dismissal:', err);
      }
    }
    setSystemNotice(null);
  };

  // ─────────────────── App data ───────────────────
  const bootstrapApp = async () => {
    try {
      const health = await api.checkHealth();
      setBackendOnline(health?.status === 'healthy' || health?.status === 'ok');
    } catch (err) {
      console.warn('Backend health check error:', err);
      setBackendOnline(false);
      return;
    }

    try {
      const quota = await api.getStorageQuota();
      setStorageQuota(quota);
    } catch (err) {
      console.warn('Storage quota fetch error:', err);
    }

    try {
      const dsList = await api.listDatasets();
      const datasetsArr = Array.isArray(dsList) ? dsList : (dsList?.datasets || []);
      setDatasets(datasetsArr);
    } catch (err) {
      console.warn('Datasets fetch error:', err);
    }

    try {
      const exps = await api.listExperiments();
      const expsArr = Array.isArray(exps) ? exps : (exps?.experiments || []);
      setExperiments(expsArr);
      if (expsArr.length > 0 && !activeExperiment) {
        setActiveExperiment(expsArr[0]);
      }
    } catch (err) {
      console.warn('Experiments fetch error:', err);
    }

    try {
      const adm = await api.checkAdminStatus();
      setIsAdmin(!!adm?.is_admin);
    } catch (err) {
      console.warn('Admin check error:', err);
    }
  };

  const refreshQuotaAndDatasets = async () => {
    try {
      const quota = await api.getStorageQuota();
      setStorageQuota(quota);
    } catch (err) {
      console.warn('Refresh quota error:', err);
    }
    try {
      const dsList = await api.listDatasets();
      const datasetsArr = Array.isArray(dsList) ? dsList : (dsList?.datasets || []);
      setDatasets(datasetsArr);
    } catch (err) {
      console.warn('Refresh error:', err);
    }
  };

  const refreshExperiments = async () => {
    try {
      const exps = await api.listExperiments();
      setExperiments(exps || []);
    } catch (err) {
      console.error('Refresh experiments error:', err);
    }
  };

  const handleSelectDataset = (ds) => {
    setActiveDataset(ds);
    setCurrentTab('dataset-lab');
  };

  const handleDatasetUploaded = (newDs) => {
    setActiveDataset(newDs);
    refreshQuotaAndDatasets();
  };

  const handleDatasetDeleted = (deletedId) => {
    if (activeDataset?.dataset_id === deletedId) {
      setActiveDataset(null);
    }
    refreshQuotaAndDatasets();
  };

  const handleProceedToStudio = (ds, target, detectedTask) => {
    setActiveDataset(ds);
    setTargetColumn(target);
    setTaskType(detectedTask);
    setCurrentTab('experiment-studio');
  };

  const handleExperimentCompleted = (completedExp) => {
    setIsBenchmarking(false);
    setActiveExperiment(completedExp);
    refreshExperiments();
    refreshQuotaAndDatasets();
    setCurrentTab('results');
  };

  const handleSelectExperiment = async (expId) => {
    try {
      const exp = await api.getExperiment(expId);
      setActiveExperiment(exp);
      setCurrentTab('results');
    } catch (err) {
      alert(`Failed to load experiment: ${err.message}`);
    }
  };

  const handleExperimentDeleted = (expId) => {
    if (activeExperiment?.experiment_id === expId) {
      setActiveExperiment(null);
    }
    refreshExperiments();
    refreshQuotaAndDatasets();
  };

  const getHeaderTitle = () => {
    switch (currentTab) {
      case 'dashboard': return 'Overview & Workspaces';
      case 'dataset-lab': return 'Dataset Lab & 8-Point Quality Audit';
      case 'experiment-studio': return 'Experiment Studio & Model Selection';
      case 'results': return 'Benchmark Results & Model Diagnostics';
      case 'history': return 'Execution History & Artifact Audit';
      case 'how-it-works': return 'How URQEN Works · Architecture & Methodology';
      case 'admin': return 'Founder Admin Console · Multi-Tenant Governance';
      default: return 'URQEN Tabular Studio';
    }
  };

  // ─────────────────── Render ───────────────────

  // Keep the branded launch screen visible while the initial auth check completes.
  if (authLoading || !launchMinimumElapsed) {
    return (
      <main className="urqen-launch-screen" role="status" aria-label="URQEN is starting">
        <div className="urqen-launch-grid" aria-hidden="true" />
        <div className="urqen-launch-content">
          <div className="urqen-launch-orbit" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div className="urqen-launch-brand">
            <UrqenLogo size={76} wordmarkSize={36} animated />
          </div>
          <div className="urqen-launch-caption">
            <span className="urqen-launch-indicator" aria-hidden="true" />
            Preparing your workspace
          </div>
          <div className="urqen-launch-progress" aria-hidden="true">
            <span />
          </div>
        </div>
      </main>
    );
  }

  // Not signed in → show Auth screen
  if (!session) {
    return (
      <>
        <AuthView
          onAuthSuccess={handleAuthSuccess}
          onOpenLegal={handleOpenLegal}
        />
        <LegalModal
          isOpen={legalModalOpen}
          onClose={() => setLegalModalOpen(false)}
          initialTab={legalModalTab}
        />
      </>
    );
  }

  // Signed in → show main app
  return (
    <div className="app-container">
      <Sidebar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        storageQuota={storageQuota}
        backendOnline={backendOnline}
        hasActiveExperiment={!!activeExperiment}
        isAdmin={isAdmin}
      />

      <div className="main-wrapper">
        <Header
          title={getHeaderTitle()}
          activeDataset={activeDataset}
          mode={mode}
          setMode={setMode}
          isBenchmarking={isBenchmarking}
          user={user}
          onSignOut={handleSignOut}
        />

        {maintenance.active && (
          <div
            style={{
              background: 'linear-gradient(90deg, #b45309 0%, #d97706 100%)',
              color: '#ffffff',
              padding: '10px 20px',
              fontSize: '13px',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              boxShadow: '0 2px 10px rgba(217, 119, 6, 0.3)',
              position: 'relative',
              zIndex: 90,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: '16px' }}>⚠️</span>
              <span>{maintenance.message}</span>
            </div>
            {isAdmin && (
              <button
                onClick={() => setCurrentTab('admin')}
                style={{
                  background: 'rgba(255, 255, 255, 0.2)',
                  border: '1px solid rgba(255, 255, 255, 0.4)',
                  color: '#fff',
                  borderRadius: 4,
                  padding: '4px 10px',
                  fontSize: 11,
                  cursor: 'pointer',
                  fontWeight: 700,
                  whiteSpace: 'nowrap',
                }}
              >
                Founder Console Controls
              </button>
            )}
          </div>
        )}

        {systemNotice && (
          <div
            role="presentation"
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 10000,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 20,
              background: 'rgba(2, 6, 23, 0.78)',
              backdropFilter: 'blur(8px)',
            }}
          >
            <section
              role="dialog"
              aria-modal="true"
              aria-labelledby="system-notice-title"
              style={{
                width: '100%',
                maxWidth: 520,
                padding: 28,
                borderRadius: 18,
                border: '1px solid var(--border-color)',
                background: 'var(--bg-card)',
                boxShadow: 'var(--shadow-xl)',
              }}
            >
              <div style={{ color: 'var(--primary)', fontSize: 12, fontWeight: 800, letterSpacing: 2, marginBottom: 12 }}>
                URQEN ACCOUNT NOTICE
              </div>
              <h2 id="system-notice-title" style={{ margin: '0 0 12px', color: 'var(--text-main)', fontSize: 23 }}>
                {systemNotice.title || 'Your URQEN history was cleared'}
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: 14, lineHeight: 1.7, margin: '0 0 14px' }}>
                {systemNotice.message}
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: 14, lineHeight: 1.7, margin: '0 0 22px' }}>
                Upgrade to URQEN Pro for more storage and a better experience.
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'flex-end' }}>
                <button className="btn btn-secondary" onClick={dismissSystemNotice}>
                  Got it
                </button>
                <button
                  className="btn btn-primary"
                  onClick={() => window.alert('This version is not yet released and will be released soon. Thank You')}
                >
                  Upgrade
                </button>
              </div>
            </section>
          </div>
        )}

        {currentTab === 'dashboard' && (
          <DashboardView
            datasets={datasets}
            experiments={experiments}
            onNavigate={setCurrentTab}
            onSelectDataset={handleSelectDataset}
            onSelectExperiment={handleSelectExperiment}
          />
        )}

        {currentTab === 'dataset-lab' && (
          <DatasetLabView
            datasets={datasets}
            activeDataset={activeDataset}
            onSelectDataset={setActiveDataset}
            onDatasetDeleted={handleDatasetDeleted}
            onDatasetUploaded={handleDatasetUploaded}
            onProceedToStudio={handleProceedToStudio}
          />
        )}

        {currentTab === 'experiment-studio' && (
          <ExperimentStudioView
            activeDataset={activeDataset}
            targetColumn={targetColumn}
            initialTaskType={taskType}
            mode={mode}
            setMode={setMode}
            onExperimentCompleted={handleExperimentCompleted}
            isBenchmarking={isBenchmarking}
            setIsBenchmarking={setIsBenchmarking}
            maintenanceMode={maintenance.active}
          />
        )}

        {currentTab === 'results' && (
          <ResultsView
            experiment={activeExperiment}
            onSelectAnotherExperiment={() => setCurrentTab('history')}
          />
        )}

        {currentTab === 'history' && (
          <HistoryView
            experiments={experiments}
            onSelectExperiment={handleSelectExperiment}
            onExperimentDeleted={handleExperimentDeleted}
          />
        )}

        {currentTab === 'how-it-works' && (
          <HowItWorksView onNavigate={setCurrentTab} onOpenLegal={handleOpenLegal} />
        )}

        {currentTab === 'admin' && (
          <AdminView currentUserId={user?.id} />
        )}

        {/* 404 fallback — unknown tab */}
        {!['dashboard', 'dataset-lab', 'experiment-studio', 'results', 'history', 'how-it-works', 'admin'].includes(currentTab) && (
          <NotFoundView onNavigate={setCurrentTab} />
        )}

        {currentTab !== 'how-it-works' && (
          <Footer onOpenLegal={handleOpenLegal} />
        )}
      </div>

      <LegalModal
        isOpen={legalModalOpen}
        onClose={() => setLegalModalOpen(false)}
        initialTab={legalModalTab}
      />
    </div>
  );
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught error:', error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 40, fontFamily: 'sans-serif', maxWidth: 600, margin: '60px auto', textAlign: 'center' }}>
          <h2 style={{ color: '#ef4444', marginBottom: 12 }}>URQEN encountered an issue</h2>
          <p style={{ color: '#64748b', fontSize: 14, marginBottom: 20 }}>
            {this.state.error?.message || 'An unexpected rendering error occurred.'}
          </p>
          <button
            onClick={() => window.location.reload()}
            style={{ padding: '10px 20px', backgroundColor: '#4f46e5', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600 }}
          >
            Reload Studio
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <ErrorBoundary>
      <UrqenApp />
    </ErrorBoundary>
  );
}
