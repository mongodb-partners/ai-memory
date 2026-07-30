import React from 'react';
import type { MemoryEvent, MemoryHit } from '../api/client';
import { useChat } from '../context/ChatContext';

/**
 * The panel the talk is about.
 *
 * Four groups, one per memory tier, each showing the hits with their rank and
 * score. The retail UI this borrows its look from joined recalled memories into
 * a single `"- {text}"` string and threw the scores away — which is exactly the
 * evidence a memory demo needs on screen. Here every number the server computed
 * is visible.
 *
 * Note on the score: `$rankFusion` returns a reciprocal-rank sum, so a
 * first-place hit scores ~1/61 ≈ 0.016. The server pre-renders a readable
 * `rank` label (`#1 · rrf 0.0164`) and this component displays that, keeping the
 * raw value in the tooltip. No rescaling — a made-up "relevance %" would be a
 * fabricated measurement on a slide about measurable retrieval.
 */

const TIERS = [
  {
    key: 'stm' as const,
    label: 'Short-term',
    blurb: 'this thread, TTL-expired',
  },
  {
    key: 'ltm' as const,
    label: 'Long-term',
    blurb: 'durable facts, importance-scored',
  },
  {
    key: 'episodic' as const,
    label: 'Episodic',
    blurb: 'what the agent did — steps, tools, files',
  },
];

const panelStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-lg)',
  overflow: 'hidden',
  minHeight: 0,
};

const headerStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: 10,
  padding: '12px 14px',
  borderBottom: '1px solid var(--border)',
  background: 'rgba(0,237,100,0.03)',
};

const titleStyle: React.CSSProperties = {
  fontFamily: 'var(--font-display)',
  fontSize: 16,
  fontWeight: 500,
  color: 'var(--text)',
};

const subtitleStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: '1px',
  color: 'var(--text-secondary)',
  marginTop: 2,
};

const bodyStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '4px 0 10px',
  minHeight: 0,
};

const groupLabelStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'baseline',
  gap: 8,
  padding: '10px 14px 6px',
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: '1px',
  color: 'var(--spring-green)',
};

const groupBlurbStyle: React.CSSProperties = {
  textTransform: 'none',
  letterSpacing: 0,
  color: 'var(--text-secondary)',
  fontSize: 10,
};

const hitStyle: React.CSSProperties = {
  margin: '0 12px 6px',
  padding: '8px 10px',
  borderRadius: 'var(--radius-sm)',
  background: 'rgba(255,255,255,0.02)',
  border: '1px solid var(--border)',
  animation: 'memory-land 1.4s ease-out',
};

const hitTextStyle: React.CSSProperties = {
  fontSize: 12.5,
  lineHeight: 1.5,
  color: 'var(--text)',
  wordBreak: 'break-word',
};

const metaRowStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  alignItems: 'center',
  gap: 8,
  marginTop: 6,
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  color: 'var(--text-secondary)',
};

const rankStyle: React.CSSProperties = {
  color: 'var(--spring-green)',
  fontWeight: 600,
};

const emptyStyle: React.CSSProperties = {
  margin: '0 14px 4px',
  fontSize: 11.5,
  color: 'var(--text-secondary)',
  fontStyle: 'italic',
};

function chipStyle(tone: 'neutral' | 'green' | 'warn'): React.CSSProperties {
  const map = {
    neutral: ['rgba(255,255,255,0.05)', 'var(--border)', 'var(--text-secondary)'],
    green: ['var(--green-tint-12)', 'var(--green-border)', 'var(--spring-green)'],
    warn: ['var(--warning-tint)', 'var(--warning-border)', 'var(--warning)'],
  } as const;
  const [background, borderColor, color] = map[tone];
  return {
    background,
    border: `1px solid ${borderColor}`,
    color,
    borderRadius: 'var(--radius-pill)',
    padding: '2px 8px',
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    letterSpacing: '0.5px',
    whiteSpace: 'nowrap',
  };
}

/** Format an ISO timestamp compactly, tolerating whatever the server sent. */
function shortTime(ts?: string | null): string | null {
  if (!ts) return null;
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function HitRow({ hit }: { hit: MemoryHit }) {
  const when = shortTime(hit.ts);
  return (
    <div style={hitStyle}>
      <div style={hitTextStyle}>{hit.text || <em>(no text)</em>}</div>
      <div style={metaRowStyle}>
        {hit.rank && (
          <span
            style={rankStyle}
            title={
              hit.score == null
                ? 'no score (listed, not searched)'
                : `raw fused score ${hit.score}`
            }
          >
            {hit.rank}
          </span>
        )}
        {/* `!= null` on purpose: importance 0 is a real, meaningful score. */}
        {hit.importance != null && <span>importance {hit.importance.toFixed(2)}</span>}
        {hit.access_count != null && <span>reads {hit.access_count}</span>}
        {hit.step != null && <span>step {hit.step}</span>}
        {hit.tools && hit.tools.length > 0 && (
          <span style={{ color: 'var(--spring-green)' }}>
            tools: {hit.tools.join(', ')}
          </span>
        )}
        {hit.files && hit.files.length > 0 && <span>files: {hit.files.join(', ')}</span>}
        {when && <span>{when}</span>}
      </div>
    </div>
  );
}

function CacheRow({
  cache,
}: {
  cache: {
    hit: boolean;
    score?: number | null;
    replayed?: boolean;
    match?: string | null;
  };
}) {
  return (
    <div style={hitStyle}>
      <div style={{ ...metaRowStyle, marginTop: 0 }}>
        <span style={chipStyle(cache.hit ? 'green' : 'neutral')}>
          {cache.hit ? 'HIT' : 'MISS'}
        </span>
        {/* Which path answered. An exact match skipped the embedding call too, so
         *  calling it "similarity 1.000" without qualification would overstate
         *  what the vector index did on this particular turn. */}
        {cache.hit && cache.match === 'exact' && <span>exact match</span>}
        {cache.hit && cache.match !== 'exact' && cache.score != null && (
          <span>similarity {cache.score.toFixed(3)}</span>
        )}
        <span>
          {cache.hit
            ? 'answer replayed from Atlas — no model call'
            : 'no near-duplicate question; the model ran'}
        </span>
      </div>
    </div>
  );
}

function WriteRow({ ev }: { ev: MemoryEvent }) {
  const failed = Boolean(ev.error);
  return (
    <div style={hitStyle}>
      <div style={{ ...metaRowStyle, marginTop: 0 }}>
        <span style={chipStyle(failed ? 'warn' : 'green')}>
          {failed ? 'WRITE FAILED' : 'WROTE'}
        </span>
        <span style={{ textTransform: 'uppercase' }}>{ev.tier}</span>
        <span style={{ color: failed ? 'var(--warning)' : 'var(--text-secondary)' }}>
          {ev.error || ev.note || 'persisted'}
        </span>
      </div>
    </div>
  );
}

const MemorySection: React.FC = () => {
  const { memory, memoryEnabled, browsing, refreshMemories, isLoading } = useChat();

  const total =
    memory.recall.stm.length +
    memory.recall.ltm.length +
    memory.recall.episodic.length;

  return (
    <section style={panelStyle} className="memory-panel" aria-label="Agent memory">
      <div style={headerStyle}>
        <div>
          <div style={titleStyle}>Memory</div>
          <div style={subtitleStyle}>
            {memoryEnabled
              ? browsing
                ? 'stored documents'
                : `recalled this turn · ${total}`
              : 'disabled for this turn'}
          </div>
        </div>
        <button
          type="button"
          onClick={refreshMemories}
          disabled={isLoading}
          title="Read the stored documents straight from Atlas, without a turn"
          style={{
            ...chipStyle('neutral'),
            cursor: isLoading ? 'default' : 'pointer',
            opacity: isLoading ? 0.5 : 1,
            padding: '5px 10px',
          }}
        >
          Browse Atlas
        </button>
      </div>

      <div style={bodyStyle}>
        {!memoryEnabled && (
          <p style={{ ...emptyStyle, margin: '12px 14px' }}>
            Memory is off. No recall runs, the semantic cache is bypassed, and
            nothing is written back — the model sees only this thread&apos;s last
            two turns.
          </p>
        )}

        {memoryEnabled && (
          <>
            {memory.pending && (
              <p style={{ ...emptyStyle, margin: '12px 14px' }}>
                Searching Atlas…
              </p>
            )}

            {/* Cache first, because it runs first — and because a hit means the
                tiers below were never queried at all. */}
            <div style={groupLabelStyle}>
              <span>Semantic cache</span>
              <span style={groupBlurbStyle}>same question asked before?</span>
            </div>
            {memory.cache ? (
              <CacheRow cache={memory.cache} />
            ) : (
              <p style={emptyStyle}>not consulted</p>
            )}

            {TIERS.map(({ key, label, blurb }) => {
              const hits = memory.recall[key];
              return (
                <div key={key}>
                  <div style={groupLabelStyle}>
                    <span>
                      {label} · {hits.length}
                    </span>
                    <span style={groupBlurbStyle}>{blurb}</span>
                  </div>
                  {hits.length === 0 ? (
                    <p style={emptyStyle}>nothing recalled</p>
                  ) : (
                    hits.map((hit, i) => <HitRow key={`${key}-${i}`} hit={hit} />)
                  )}
                </div>
              );
            })}

            {memory.writes.length > 0 && (
              <>
                <div style={groupLabelStyle}>
                  <span>Written back</span>
                  <span style={groupBlurbStyle}>after the answer</span>
                </div>
                {memory.writes.map((ev, i) => (
                  <WriteRow key={`w-${i}`} ev={ev} />
                ))}
              </>
            )}
          </>
        )}
      </div>
    </section>
  );
};

export default MemorySection;
