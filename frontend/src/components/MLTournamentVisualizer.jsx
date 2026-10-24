import { useEffect, useState } from 'react';
import { ArrowRight, BarChart3, Database, ScanSearch, SlidersHorizontal } from 'lucide-react';

const WORKFLOW_STEPS = [
  { label: 'Your data', detail: 'Upload a tabular file', icon: Database },
  { label: 'Inspect', detail: 'Review profile and quality', icon: ScanSearch },
  { label: 'Configure', detail: 'Choose a target and task', icon: SlidersHorizontal },
  { label: 'Evaluate', detail: 'Compare experiment results', icon: BarChart3 },
];

export default function MLTournamentVisualizer() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined;

    const timer = window.setInterval(() => {
      setActiveStep((step) => (step + 1) % WORKFLOW_STEPS.length);
    }, 1800);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <section className="workflow-visual" aria-label="URQEN workflow overview">
      <div className="workflow-visual-heading">
        <div>
          <div className="workflow-eyebrow">A clear path from data to results</div>
          <p>See what happens at each stage. You stay in control of the experiment.</p>
        </div>
        <span className="workflow-step-counter" aria-live="polite">
          STEP {String(activeStep + 1).padStart(2, '0')} / {String(WORKFLOW_STEPS.length).padStart(2, '0')}
        </span>
      </div>

      <div className="workflow-steps">
        {WORKFLOW_STEPS.map(({ label, detail, icon: Icon }, index) => (
          <div
            key={label}
            className={`workflow-step${index === activeStep ? ' is-active' : ''}${index < activeStep ? ' is-complete' : ''}`}
            aria-current={index === activeStep ? 'step' : undefined}
          >
            <div className="workflow-step-marker">
              <Icon size={17} strokeWidth={1.8} />
            </div>
            <div className="workflow-step-copy">
              <span className="workflow-step-label">{label}</span>
              <span className="workflow-step-detail">{detail}</span>
            </div>
            {index < WORKFLOW_STEPS.length - 1 && (
              <ArrowRight className="workflow-step-arrow" size={14} aria-hidden="true" />
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
