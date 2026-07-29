import React, { useState } from 'react';
import { useChat } from '../context/ChatContext';

export interface Preset {
  icon: string;
  text: string;
  /** Why this prompt is here, shown under the card. The demo is an argument,
   *  and each prompt is one step in it. */
  proves: string;
  /** Start a fresh thread. Cross-thread recall is the point of the demo, and it
   *  is only proved if the thread the answer arrives in is demonstrably empty. */
  newThread?: boolean;
}

/**
 * The prompts from `docs/talks/ai4-2026/demo-script.md`, verbatim.
 *
 * They are kept identical to the script on purpose: the two questions must match
 * character-for-character between the memory-OFF and memory-ON passes, or the
 * comparison is not a comparison. Retyping them live is how a wording drift
 * sneaks in and undermines the claim in front of the audience.
 */
export const PRESETS: Preset[] = [
  {
    icon: '1',
    text: "I'm allergic to shellfish, and I'm cooking for six people on Friday.",
    proves: 'Writes two facts. Watch the panel’s write-back rows.',
  },
  {
    icon: '2',
    text: 'What should I make Friday?',
    proves: 'Asked in a fresh thread. Memory off: no idea. Memory on: recalls both facts.',
    newThread: true,
  },
  {
    icon: '3',
    text: 'What should I make Friday?',
    proves:
      'The same question again — a semantic cache HIT. Answered with no model call.',
  },
  {
    icon: '4',
    text: 'What have we worked on together so far?',
    proves: 'Episodic recall: what the agent did, not what it knows.',
    newThread: true,
  },
];

const sectionStyle: React.CSSProperties = {
  padding: '14px 16px 4px',
};

const headingStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: '1.5px',
  color: 'var(--spring-green)',
  marginBottom: 10,
};

const gridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 240px), 1fr))',
  gap: 10,
};

const baseCardStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 6,
  padding: '11px 13px',
  background: 'var(--surface-2)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)',
  cursor: 'pointer',
  textAlign: 'left',
  color: 'var(--text)',
  transition: 'transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease',
};

const stepStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  letterSpacing: '1px',
  color: 'var(--spring-green)',
  display: 'flex',
  alignItems: 'center',
  gap: 8,
};

const cardTextStyle: React.CSSProperties = {
  fontSize: 13,
  lineHeight: 1.5,
  color: 'var(--text)',
};

const provesStyle: React.CSSProperties = {
  fontSize: 11,
  lineHeight: 1.45,
  color: 'var(--text-secondary)',
};

function PresetCard({
  preset,
  index,
  onAsk,
  disabled,
}: {
  preset: Preset;
  index: number;
  onAsk: () => void;
  disabled: boolean;
}) {
  const [hover, setHover] = useState(false);
  const style: React.CSSProperties =
    hover && !disabled
      ? {
          ...baseCardStyle,
          transform: 'translateY(-2px)',
          borderColor: 'var(--green-border)',
          boxShadow: '0 0 0 1px rgba(0,237,100,0.15), 0 10px 24px rgba(0,0,0,0.35)',
        }
      : { ...baseCardStyle, opacity: disabled ? 0.55 : 1, cursor: disabled ? 'default' : 'pointer' };
  return (
    <button
      type="button"
      style={style}
      onClick={onAsk}
      disabled={disabled}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={() => setHover(false)}
    >
      <span style={stepStyle}>
        <span>STEP {index + 1}</span>
        {preset.newThread && (
          <span style={{ color: 'var(--text-secondary)' }}>· new thread</span>
        )}
      </span>
      <span style={cardTextStyle}>{preset.text}</span>
      <span style={provesStyle}>{preset.proves}</span>
    </button>
  );
}

export default function PresetGrid() {
  const { sendMessage, isLoading } = useChat();
  return (
    <section style={sectionStyle} id="presets">
      <div style={headingStyle}>The demo, in four clicks</div>
      <div style={gridStyle}>
        {PRESETS.map((p, i) => (
          <PresetCard
            key={`${i}-${p.text}`}
            preset={p}
            index={i}
            disabled={isLoading}
            onAsk={() =>
              sendMessage(p.text, p.newThread ? { newThread: true } : undefined)
            }
          />
        ))}
      </div>
    </section>
  );
}
