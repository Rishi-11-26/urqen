import React from 'react';

/**
 * UrqenLogo
 * Modern geometric vector brand emblem representing multifaceted tabular intelligence,
 * data flow, and empirical modeling.
 */
export default function UrqenLogo({ size = 32, showWordmark = true, wordmarkSize = 16, animated = false }) {
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 10,
        userSelect: 'none',
      }}
      className={animated ? 'urqen-logo-animated' : ''}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 48 48"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="urqen-mark"
        style={{
          filter: 'drop-shadow(0 4px 12px var(--primary-mid))',
          flexShrink: 0,
        }}
      >
        <defs>
          <linearGradient id="urqenMarkFrame" x1="4" y1="4" x2="44" y2="44" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="var(--primary)" />
            <stop offset="100%" stopColor="var(--accent-blue)" />
          </linearGradient>
          <linearGradient id="urqenMarkLine" x1="10" y1="36" x2="37" y2="12" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="var(--primary)" />
            <stop offset="100%" stopColor="var(--accent-blue)" />
          </linearGradient>
        </defs>
        <rect
          x="4.75"
          y="4.75"
          width="38.5"
          height="38.5"
          rx="13"
          fill="var(--bg-subtle)"
          stroke="url(#urqenMarkFrame)"
          strokeWidth="1.5"
        />
        <path
          d="M11 34.5L19 26L25.5 29L36.5 14"
          stroke="url(#urqenMarkLine)"
          strokeWidth="2.7"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path d="M11 14V35M11 35H37" stroke="var(--border-color)" strokeWidth="1" />
        <circle cx="11" cy="34.5" r="2.2" fill="var(--primary)" />
        <circle cx="19" cy="26" r="2.2" fill="var(--primary)" />
        <circle cx="25.5" cy="29" r="2.2" fill="var(--accent-blue)" />
        <circle cx="36.5" cy="14" r="2.7" fill="var(--primary)" className="urqen-mark-point" />
      </svg>

      {showWordmark && (
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div
            style={{
              fontSize: wordmarkSize,
              fontWeight: 800,
              letterSpacing: '-0.03em',
              lineHeight: 1.05,
              background: 'var(--logo-wordmark-gradient)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              fontFamily: "'Syne', var(--font-sans)",
            }}
          >
            URQEN
          </div>
          <span
            style={{
              fontSize: 9,
              fontWeight: 700,
              color: 'var(--primary)',
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              marginTop: 1,
              fontFamily: 'var(--font-mono)',
            }}
          >
            STUDIO // ML
          </span>
        </div>
      )}
    </div>
  );
}
