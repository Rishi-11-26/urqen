import { useState } from 'react';
import { ChevronDown, ChevronRight, CheckCircle, AlertTriangle, AlertCircle, Info } from 'lucide-react';

const SEVERITY_MAP = {
  high:    { cls: 'fail',    Icon: AlertCircle,   color: 'var(--danger)',  label: 'High Risk' },
  fail:    { cls: 'fail',    Icon: AlertCircle,   color: 'var(--danger)',  label: 'High Risk' },
  medium:  { cls: 'warning', Icon: AlertTriangle, color: 'var(--warning)', label: 'Warning' },
  warning: { cls: 'warning', Icon: AlertTriangle, color: 'var(--warning)', label: 'Warning' },
  low:     { cls: 'info',    Icon: Info,          color: 'var(--info)',    label: 'Low' },
  pass:    { cls: 'pass',    Icon: CheckCircle,   color: 'var(--success)', label: 'Passed' },
  info:    { cls: 'info',    Icon: Info,          color: 'var(--info)',    label: 'Info' },
};

export default function QualityFindingCard({ finding }) {
  const [expanded, setExpanded] = useState(false);

  if (!finding) return null;

  const severityKey = (finding.severity || finding.status || 'info').toLowerCase();
  const { cls, Icon, color, label } = SEVERITY_MAP[severityKey] || SEVERITY_MAP.info;

  const title = finding.issue || finding.check || finding.name || 'Quality Inspection';
  const column = finding.affected_column || finding.column;
  const evidence = finding.evidence || finding.message || finding.description;
  const whyItMatters = finding.why_it_matters;
  const recommendedAction = finding.recommended_action || finding.recommendation;
  const tradeOffs = finding.trade_offs;

  return (
    <div className="finding-card">
      <div className="finding-header" onClick={() => setExpanded((v) => !v)}>
        <span className={`finding-indicator ${cls}`} />
        <Icon size={15} color={color} style={{ flexShrink: 0 }} />

        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, flexWrap: 'wrap' }}>
          <span className="finding-name" style={{ fontWeight: 700, fontSize: 13 }}>
            {title}
          </span>
          {column && (
            <code style={{ fontSize: 11, padding: '1px 6px', color: 'var(--text-main)', background: 'var(--bg-subtle)' }}>
              {column}
            </code>
          )}
        </div>

        <span className={`badge badge-${severityKey === 'high' || severityKey === 'fail' ? 'danger' : severityKey === 'medium' || severityKey === 'warning' ? 'warning' : 'neutral'}`} style={{ fontSize: 10.5, textTransform: 'capitalize' }}>
          {label}
        </span>

        {expanded ? <ChevronDown size={14} color="var(--text-subtle)" /> : <ChevronRight size={14} color="var(--text-subtle)" />}
      </div>

      {expanded && (
        <div className="finding-body">
          {evidence && (
            <div style={{ marginBottom: 8 }}>
              <strong style={{ color: 'var(--text-main)', fontSize: 12 }}>Observed Evidence: </strong>
              <span>{evidence}</span>
            </div>
          )}

          {whyItMatters && (
            <div style={{ marginBottom: 8, color: 'var(--text-muted)' }}>
              <strong style={{ color: 'var(--text-main)', fontSize: 12 }}>Impact on Modeling: </strong>
              <span>{whyItMatters}</span>
            </div>
          )}

          {recommendedAction && (
            <div style={{ marginTop: 8, padding: '8px 12px', background: 'var(--primary-light)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--primary-mid)' }}>
              <strong style={{ color: 'var(--primary)', fontSize: 12 }}>Recommended Action: </strong>
              <span style={{ color: 'var(--text-main)' }}>{recommendedAction}</span>
            </div>
          )}

          {tradeOffs && tradeOffs !== 'None documented.' && (
            <div style={{ marginTop: 6, fontSize: 11.5, color: 'var(--text-muted)' }}>
              <em>Trade-offs:</em> {tradeOffs}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
