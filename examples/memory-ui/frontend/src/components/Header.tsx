import React from 'react';
import { useChat } from '../context/ChatContext';

const headerStyle: React.CSSProperties = {
  position: 'sticky',
  top: 0,
  zIndex: 100,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  minHeight: 56,
  padding: '0 20px',
  gap: 16,
  background: 'rgba(6,10,15,0.8)',
  backdropFilter: 'blur(12px)',
  WebkitBackdropFilter: 'blur(12px)',
  borderBottom: '1px solid var(--border)',
  flexWrap: 'wrap',
};

const leftStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  minWidth: 0,
};

const titleStyle: React.CSSProperties = {
  fontFamily: 'var(--font-display)',
  fontSize: 17,
  fontWeight: 500,
  color: 'var(--text)',
  lineHeight: 1.2,
  whiteSpace: 'nowrap',
};

const tagStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10.5,
  color: 'var(--text-secondary)',
  letterSpacing: '0.3px',
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
};

const rightStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  flexShrink: 0,
  flexWrap: 'wrap',
};

const badgeStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 7,
  padding: '5px 11px',
  background: 'var(--green-tint)',
  border: '1px solid var(--green-border)',
  borderRadius: 'var(--radius-pill)',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  color: 'var(--text)',
  maxWidth: 340,
};

const userWrapStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  padding: '4px 10px',
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-pill)',
};

const userLabelStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: '1px',
  color: 'var(--text-secondary)',
};

const userInputStyle: React.CSSProperties = {
  background: 'transparent',
  border: 'none',
  outline: 'none',
  color: 'var(--text)',
  fontFamily: 'var(--font-mono)',
  fontSize: 12,
  width: 150,
};

const iconBtnStyle: React.CSSProperties = {
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-pill)',
  color: 'var(--text-secondary)',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  padding: '6px 12px',
  cursor: 'pointer',
};

function MongoLeaf({ size = 26 }: { size?: number }) {
  return (
    <svg
      width={(size * 120) / 278}
      height={size}
      viewBox="0 0 120 278"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      style={{ flexShrink: 0, filter: 'drop-shadow(0 0 8px rgba(0,237,100,0.4))' }}
    >
      <path
        d="M82.3229 28.6444C71.5367 15.8469 62.2485 2.84945 60.351 0.149971C60.1512 -0.0499903 59.8515 -0.0499903 59.6518 0.149971C57.7542 2.84945 48.4661 15.8469 37.6798 28.6444C-54.9019 146.721 52.2613 226.406 52.2613 226.406L53.1601 227.006C53.959 239.303 55.9565 257 55.9565 257H59.9514H63.9463C63.9463 257 65.9438 239.403 66.7428 227.006L67.6416 226.306C67.7414 226.406 174.905 146.721 82.3229 28.6444ZM59.9514 224.606C59.9514 224.606 55.1576 220.507 53.8592 218.408V218.207L59.6518 89.6326C59.6518 89.2326 60.2511 89.2326 60.2511 89.6326L66.0436 218.207V218.408C64.7453 220.507 59.9514 224.606 59.9514 224.606Z"
        fill="#00ED64"
      />
    </svg>
  );
}

/**
 * The memory toggle. This is the control the whole talk turns on, so it is a
 * real switch with an unambiguous label rather than a subtle icon — from ten
 * feet away, in a crowd, the audience has to be able to read which state it is
 * in without the presenter narrating it.
 */
function MemoryToggle() {
  const { memoryEnabled, setMemoryEnabled, isLoading } = useChat();
  const on = memoryEnabled;
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={isLoading}
      onClick={() => setMemoryEnabled(!on)}
      title={
        on
          ? 'Memory ON — recall, semantic cache and write-back all active'
          : 'Memory OFF — no recall, cache bypassed, nothing written'
      }
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        padding: '6px 13px 6px 10px',
        borderRadius: 'var(--radius-pill)',
        border: `1px solid ${on ? 'var(--green-border)' : 'var(--warning-border)'}`,
        background: on ? 'var(--green-tint-12)' : 'var(--warning-tint)',
        color: on ? 'var(--spring-green)' : 'var(--warning)',
        fontFamily: 'var(--font-mono)',
        fontSize: 11.5,
        fontWeight: 600,
        letterSpacing: '0.5px',
        cursor: isLoading ? 'default' : 'pointer',
        opacity: isLoading ? 0.55 : 1,
        boxShadow: on ? 'var(--green-glow)' : 'none',
        transition: 'background 160ms, border-color 160ms, color 160ms',
      }}
    >
      <span
        aria-hidden="true"
        style={{
          width: 26,
          height: 14,
          borderRadius: 'var(--radius-pill)',
          background: on ? 'rgba(0,237,100,0.3)' : 'rgba(255,192,16,0.25)',
          position: 'relative',
          flexShrink: 0,
        }}
      >
        <span
          style={{
            position: 'absolute',
            top: 2,
            left: on ? 14 : 2,
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: 'currentColor',
            transition: 'left 160ms',
          }}
        />
      </span>
      MEMORY {on ? 'ON' : 'OFF'}
    </button>
  );
}

/** The model and embedding actually configured server-side, read from /config. */
function ModelBadge() {
  const { config } = useChat();
  const label = config
    ? `${config.llm_model} · ${config.embedding_model} (${config.embedding_dimension}d)`
    : 'loading…';
  return (
    <div
      style={badgeStyle}
      title={
        config
          ? `LLM via ${config.llm_provider}; embeddings via ${config.embedding_provider}; database ${config.database}`
          : 'reading /config'
      }
    >
      <svg width="9" height="9" viewBox="0 0 16 16" fill="var(--spring-green)">
        <circle cx="8" cy="8" r="5" />
      </svg>
      <span
        style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
      >
        {label}
      </span>
    </div>
  );
}

export default function Header() {
  const { userId, setUserId, newConversation, wipeUser } = useChat();

  // A wipe is unrecoverable and sits next to the buttons used mid-demo, so it
  // asks first. Losing the seeded data between the two booth mornings would
  // cost a rehearsal, not just a click.
  const confirmWipe = () => {
    const ok = window.confirm(
      `Delete every memory for “${userId}”? This cannot be undone.`,
    );
    if (ok) void wipeUser();
  };

  return (
    <header style={headerStyle}>
      <div style={leftStyle}>
        <MongoLeaf size={26} />
        <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <span style={titleStyle}>Give Your Agents a Memory</span>
          <span style={tagStyle}>
            MongoDB Atlas · short-term · long-term · episodic · semantic cache
          </span>
        </div>
      </div>
      <div style={rightStyle}>
        <MemoryToggle />
        <ModelBadge />
        {/* Switching the user id is how per-user isolation gets proved: the same
            question, a different user, and the memories do not cross. */}
        <label style={userWrapStyle} title="Memories are scoped to this user id">
          <span style={userLabelStyle}>User</span>
          <input
            style={userInputStyle}
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="ai4-demo"
            aria-label="User ID"
            spellCheck={false}
          />
        </label>
        <button type="button" style={iconBtnStyle} onClick={newConversation}>
          New thread
        </button>
        <button
          type="button"
          style={{ ...iconBtnStyle, color: 'var(--danger)' }}
          onClick={confirmWipe}
        >
          Wipe
        </button>
      </div>
    </header>
  );
}
