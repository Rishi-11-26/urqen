import { useState } from 'react';
import { supabase } from '../lib/supabase';
import { Mail, Lock, Eye, EyeOff, AlertCircle, CheckCircle2, ArrowRight, ShieldCheck, Cpu, BarChart3, Database } from 'lucide-react';
import UrqenLogo from '../components/UrqenLogo';
import MLTournamentVisualizer from '../components/MLTournamentVisualizer';

export default function AuthView({ onAuthSuccess, onOpenLegal }) {
  const [mode, setMode] = useState('login'); // 'login' | 'signup' | 'forgot'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const { data, error: err } = await supabase.auth.signInWithPassword({ email, password });
    setLoading(false);
    if (err) { setError(err.message); return; }
    onAuthSuccess(data.session, data.user);
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setError(null);
    if (!agreedToTerms) { setError('Please agree to the Terms of Service and Privacy Policy to create an account.'); return; }
    if (password !== confirmPassword) { setError('Passwords do not match.'); return; }
    if (password.length < 6) { setError('Password must be at least 6 characters.'); return; }
    setLoading(true);
    const { error: err } = await supabase.auth.signUp({ email, password });
    setLoading(false);
    if (err) { setError(err.message); return; }
    setSuccess(
      `Your URQEN account request was created. Check ${email} for the email confirmation link to activate your account. The message may currently appear to be from Supabase. If it is not in your inbox, check your spam or junk folder.`
    );
    setMode('login');
  };

  const handleForgot = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const { error: err } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password`,
    });
    setLoading(false);
    if (err) { setError(err.message); return; }
    setSuccess('Password reset link sent! Check your email.');
  };

  const inputStyle = {
    width: '100%',
    padding: '12px 14px',
    border: '1.5px solid var(--border-color)',
    borderRadius: 'var(--radius-sm)',
    fontSize: 13.5,
    background: 'var(--bg-input)',
    color: 'var(--text-main)',
    outline: 'none',
    transition: 'all var(--duration-fast)',
    boxSizing: 'border-box',
    fontFamily: 'var(--font-sans)',
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        width: '100vw',
        background: 'var(--auth-background)',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        overflowX: 'hidden',
      }}
      className="auth-screen"
    >
      {/* Top Editorial Bar */}
      <header
        style={{
          padding: '24px 36px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid var(--border-light)',
          zIndex: 10,
        }}
        className="auth-topbar"
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <UrqenLogo size={36} showWordmark={true} wordmarkSize={18} animated={true} />
        </div>
      </header>

      {/* Main Showcase Hero */}
      <div
        className="auth-showcase"
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '36px 40px',
          zIndex: 2,
        }}
      >
        <div
          style={{
            maxWidth: 1240,
            width: '100%',
            display: 'grid',
            gridTemplateColumns: '1.2fr 0.9fr',
            gap: 56,
            alignItems: 'center',
          }}
          className="auth-grid-container"
        >
          {/* Left: Scientific Digital Experience & Visual Storytelling */}
          <div className="auth-story">
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '5px 12px',
                borderRadius: 20,
                background: 'var(--bg-subtle)',
                border: '1px solid var(--border-color)',
                color: 'var(--text-muted)',
                fontSize: 12,
                fontFamily: 'var(--font-mono)',
                marginBottom: 18,
              }}
            >
              <Cpu size={14} color="var(--primary)" />
              <span>                            TABULAR MACHINE LEARNING</span>
            </div>

            <h1
              style={{
                fontFamily: "'Syne', var(--font-sans)",
                fontSize: 'clamp(32px, 4.5vw, 54px)',
                fontWeight: 800,
                letterSpacing: '-0.04em',
                lineHeight: 1.08,
                color: 'var(--text-main)',
                marginBottom: 18,
              }}
            >
              Find the signal<br />
              <span>
                in your data.
              </span>
            </h1>

            <p
              style={{
                fontSize: 15,
                color: 'var(--text-muted)',
                lineHeight: 1.65,
                maxWidth: 520,
                marginBottom: 28,
              }}
            >
              A focused workspace for exploring tabular data, shaping experiments, and understanding the results.
            </p>

            {/* Interactive ML Tournament Visualizer */}
            <div style={{ marginBottom: 28 }}>
              <MLTournamentVisualizer />
            </div>

            {/* Three Scientific Pillars */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: 16,
              }}
              className="auth-pillars"
            >
              {[
                { title: 'Inspect', desc: 'Explore the shape and quality of your data.', icon: Database },
                { title: 'Configure', desc: 'Set a target, task, and experiment options.', icon: Cpu },
                { title: 'Understand', desc: 'Compare results and review available diagnostics.', icon: BarChart3 },
              ].map(({ title, desc, icon: Icon }) => (
                <div
                  key={title}
                  style={{
                    padding: '14px 16px',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border-light)',
                  }}
                >
                  <Icon size={16} color="var(--primary)" style={{ marginBottom: 6 }} />
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-main)', marginBottom: 4 }}>{title}</div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-muted)', lineHeight: 1.45 }}>{desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Sleek Authentication Console */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div
              className="card"
              data-auth-panel
              style={{
                width: '100%',
                maxWidth: 440,
                padding: 0,
                overflow: 'hidden',
                background: 'var(--bg-card)',
                border: '1px solid var(--primary-mid)',
                boxShadow: 'var(--shadow-xl), 0 0 30px var(--primary-light)',
              }}
            >
              {/* Tab Selector */}
              {mode !== 'forgot' && (
                <div
                  style={{
                    display: 'flex',
                    borderBottom: '1px solid var(--border-color)',
                    background: 'var(--bg-app)',
                  }}
                >
                  {[
                    { id: 'login', label: 'Sign In' },
                    { id: 'signup', label: 'Create Account' },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => { setMode(tab.id); setError(null); setSuccess(null); setAgreedToTerms(false); }}
                      style={{
                        flex: 1,
                        padding: '15px 0',
                        border: 'none',
                        background: 'none',
                        fontSize: 13.5,
                        fontFamily: "'Syne', var(--font-sans)",
                        fontWeight: mode === tab.id ? 700 : 600,
                        color: mode === tab.id ? 'var(--primary)' : 'var(--text-subtle)',
                        borderBottom: mode === tab.id ? '2px solid var(--primary)' : '2px solid transparent',
                        cursor: 'pointer',
                        transition: 'all var(--duration-fast)',
                        letterSpacing: '0.02em',
                      }}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              )}

              <div style={{ padding: 32 }}>
                {/* Header */}
                <div style={{ marginBottom: 24 }}>
                  <div
                    style={{
                      fontFamily: "'Syne', var(--font-sans)",
                      fontSize: 20,
                      fontWeight: 800,
                      color: 'var(--text-main)',
                      marginBottom: 6,
                    }}
                  >
                    {mode === 'login' && 'Access Your Studio'}
                    {mode === 'signup' && 'Register Workspace'}
                    {mode === 'forgot' && 'Reset Security Credentials'}
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                    {mode === 'login' && 'Sign in to access your stored datasets and trained pipelines.'}
                    {mode === 'signup' && 'Instant access to empirical tabular machine learning.'}
                    {mode === 'forgot' && "Enter your email address to receive password reset instructions."}
                  </div>
                </div>

                {/* Alerts */}
                {error && (
                  <div className="alert alert-danger" style={{ marginBottom: 18, padding: '10px 14px', fontSize: 13 }}>
                    <AlertCircle size={16} className="alert-icon" />
                    <span>{error}</span>
                  </div>
                )}
                {success && (
                  <div className="alert alert-success" role="status" aria-live="polite" style={{ marginBottom: 18, padding: '10px 14px', fontSize: 13 }}>
                    <CheckCircle2 size={16} className="alert-icon" />
                    <span>{success}</span>
                  </div>
                )}

                {/* Form */}
                <form onSubmit={mode === 'login' ? handleLogin : mode === 'signup' ? handleSignup : handleForgot}>
                  {/* Email */}
                  <div style={{ marginBottom: 16 }}>
                    <label style={{ fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Account Email
                    </label>
                    <div style={{ position: 'relative' }}>
                      <Mail size={16} style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-subtle)' }} />
                      <input
                        type="email"
                        required
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="researcher@lab.io"
                        style={{ ...inputStyle, paddingLeft: 42 }}
                        autoComplete="email"
                      />
                    </div>
                  </div>

                  {/* Password */}
                  {mode !== 'forgot' && (
                    <div style={{ marginBottom: 16 }}>
                      <label style={{ fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Password
                      </label>
                      <div style={{ position: 'relative' }}>
                        <Lock size={16} style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-subtle)' }} />
                        <input
                          type={showPassword ? 'text' : 'password'}
                          required
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          placeholder="••••••••••••"
                          style={{ ...inputStyle, paddingLeft: 42, paddingRight: 42 }}
                          autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                          minLength={6}
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword((v) => !v)}
                          style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-subtle)', padding: 0 }}
                        >
                          {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Confirm Password (signup only) */}
                  {mode === 'signup' && (
                    <div style={{ marginBottom: 16 }}>
                      <label style={{ fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Confirm Security Key
                      </label>
                      <div style={{ position: 'relative' }}>
                        <Lock size={16} style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-subtle)' }} />
                        <input
                          type={showPassword ? 'text' : 'password'}
                          required
                          value={confirmPassword}
                          onChange={(e) => setConfirmPassword(e.target.value)}
                          placeholder="Repeat password"
                          style={{ ...inputStyle, paddingLeft: 42 }}
                          autoComplete="new-password"
                          minLength={6}
                        />
                      </div>
                    </div>
                  )}

                  {/* Forgot link (login mode) */}
                  {mode === 'login' && (
                    <div style={{ textAlign: 'right', marginBottom: 20, marginTop: -6 }}>
                      <button
                        type="button"
                        onClick={() => { setMode('forgot'); setError(null); setSuccess(null); }}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 12, color: 'var(--primary)', fontWeight: 600, padding: 0 }}
                      >
                        Forgot credentials?
                      </button>
                    </div>
                  )}

                  {/* Agree to Terms (signup only) */}
                  {mode === 'signup' && (
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 18 }}>
                      <input
                        type="checkbox"
                        id="agree-terms"
                        checked={agreedToTerms}
                        onChange={(e) => setAgreedToTerms(e.target.checked)}
                        style={{ marginTop: 3, accentColor: 'var(--primary)', cursor: 'pointer', flexShrink: 0 }}
                      />
                      <label htmlFor="agree-terms" style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.55, cursor: 'pointer' }}>
                        I have read and agree to the{' '}
                        <button type="button" onClick={() => onOpenLegal?.('terms')} style={{ background: 'none', border: 'none', color: 'var(--primary)', cursor: 'pointer', fontWeight: 600, padding: 0, fontSize: 12.5, textDecoration: 'underline' }}>Terms of Service</button>
                        {' '}and{' '}
                        <button type="button" onClick={() => onOpenLegal?.('privacy')} style={{ background: 'none', border: 'none', color: 'var(--primary)', cursor: 'pointer', fontWeight: 600, padding: 0, fontSize: 12.5, textDecoration: 'underline' }}>Privacy Policy</button>
                      </label>
                    </div>
                  )}

                  {/* Submit Button */}
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={loading || (mode === 'signup' && !agreedToTerms)}
                    style={{ width: '100%', justifyContent: 'center', gap: 10, fontSize: 14, padding: '13px 0' }}
                  >
                    {loading ? (
                      <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                    ) : (
                      <ArrowRight size={16} />
                    )}
                    {loading ? 'Authenticating…' : (
                      mode === 'login' ? 'Enter Studio' :
                      mode === 'signup' ? 'Create Account & Begin' :
                      'Dispatch Reset Link'
                    )}
                  </button>

                  {/* Back to login (forgot mode) */}
                  {mode === 'forgot' && (
                    <button
                      type="button"
                      onClick={() => { setMode('login'); setError(null); setSuccess(null); }}
                      style={{ width: '100%', marginTop: 14, background: 'none', border: 'none', cursor: 'pointer', fontSize: 13, color: 'var(--text-muted)', fontWeight: 600 }}
                    >
                      ← Return to Sign In
                    </button>
                  )}
                </form>
              </div>

              {/* Data Commitment Banner */}
              <div
                style={{
                  padding: '12px 20px',
                  background: 'var(--bg-subtle)',
                  borderTop: '1px solid var(--border-light)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  fontSize: 11,
                  color: 'var(--text-muted)',
                }}
              >
                <ShieldCheck size={14} color="var(--primary)" style={{ flexShrink: 0 }} />
                <span>Zero Data-Sharing: Customer datasets are never used to train global AI models.</span>
              </div>
            </div>

            {/* Footer Links */}
            <div
              style={{
                textAlign: 'center',
                marginTop: 22,
                fontSize: 12,
                color: 'var(--text-subtle)',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                gap: 12,
              }}
            >
              <button
                type="button"
                onClick={() => onOpenLegal?.('terms')}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 12 }}
              >
                Terms of Service
              </button>
              <span>·</span>
              <button
                type="button"
                onClick={() => onOpenLegal?.('privacy')}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 12 }}
              >
                Privacy Policy
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
