import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useChat } from '../context/ChatContext';
import ChatMessage from './ChatMessage';
import PresetGrid from './PresetGrid';

/**
 * The conversation surface.
 *
 * Retail's version of this was a floating widget over a storefront, opened by a
 * FAB. Here the chat is the whole page and it sits permanently beside the memory
 * panel, because the demo's argument only lands if the audience can see the
 * answer and the retrieved memories *at the same time*. A panel that has to be
 * opened is one more thing to click on stage, and a hidden memory panel proves
 * nothing.
 */

const panelStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  background: 'var(--surface-2)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-lg)',
  boxShadow: 'var(--shadow-panel)',
  overflow: 'hidden',
  minHeight: 0,
};

const messagesStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: 16,
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
  minHeight: 0,
};

const emptyStyle: React.CSSProperties = {
  margin: 'auto',
  textAlign: 'center',
  maxWidth: 340,
  color: 'var(--text-secondary)',
  fontSize: 13,
  lineHeight: 1.7,
};

const inputStyle: React.CSSProperties = {
  flex: 1,
  padding: '11px 13px',
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)',
  color: 'var(--text)',
  fontSize: 13.5,
  outline: 'none',
};

const threadTagStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  color: 'var(--text-secondary)',
};

export default function ChatPanel() {
  const {
    messages,
    isLoading,
    memoryEnabled,
    memory,
    threadId,
    userId,
    sendMessage,
  } = useChat();
  const [draft, setDraft] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Follow the newest content as tokens stream in.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, isLoading]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = draft.trim();
    if (!text || isLoading || !userId) return;
    sendMessage(text);
    setDraft('');
  };

  const canSend = Boolean(draft.trim()) && !isLoading && Boolean(userId);
  // A cache hit answers without an inference call, so the audience should be
  // told which of the two just happened rather than inferring it from latency.
  const cacheHit = memory.cache?.hit === true;

  return (
    <section style={panelStyle} aria-label="Conversation">
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 10,
          padding: '12px 14px',
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            fontFamily: 'var(--font-mono)',
            fontSize: '0.7rem',
            textTransform: 'uppercase',
            letterSpacing: '1.2px',
            color: memoryEnabled ? 'var(--spring-green)' : 'var(--warning)',
          }}
        >
          <span aria-hidden="true">✦</span>
          Assistant · memory {memoryEnabled ? 'on' : 'off'}
        </div>
        {/* The thread id is on screen because "new thread" is a claim the
            audience should be able to check, not take on trust. */}
        <span style={threadTagStyle} title="Current thread id">
          thread {threadId.slice(0, 8)}
        </span>
      </div>

      {messages.length === 0 && <PresetGrid />}

      <div ref={scrollRef} style={messagesStyle}>
        {messages.length === 0 ? (
          <div style={emptyStyle}>
            Same model, same prompt, same code — one toggle.
            <br />
            Run the four steps above in order, or type your own question.
          </div>
        ) : (
          messages.map((m) => <ChatMessage key={m.id} message={m} />)
        )}

        {isLoading && memory.pending && (
          <div
            style={{
              alignSelf: 'flex-start',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              padding: '6px 10px',
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--spring-green)',
              background: 'var(--green-tint)',
              border: '1px solid var(--green-border)',
              borderRadius: 'var(--radius-pill)',
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: 'var(--spring-green)',
                animation: 'pulse-dot 1s ease-in-out infinite',
              }}
            />
            searching Atlas…
          </div>
        )}

        {cacheHit && (
          <div
            style={{
              alignSelf: 'flex-start',
              padding: '6px 10px',
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--spring-green)',
              background: 'var(--green-tint)',
              border: '1px solid var(--green-border)',
              borderRadius: 'var(--radius-pill)',
            }}
          >
            served from the semantic cache — no model call
          </div>
        )}
      </div>

      <form
        onSubmit={submit}
        style={{
          display: 'flex',
          gap: 8,
          padding: 12,
          borderTop: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={
            memoryEnabled
              ? 'Ask something it should remember…'
              : 'Memory is off — ask, then toggle it on and ask again'
          }
          aria-label="Message"
          spellCheck={false}
          style={inputStyle}
        />
        <button
          type="submit"
          disabled={!canSend}
          aria-label="Send message"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 44,
            background: canSend ? 'var(--spring-green)' : 'rgba(255,255,255,0.06)',
            color: canSend ? 'var(--slate-navy)' : 'var(--text-secondary)',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            cursor: canSend ? 'pointer' : 'not-allowed',
            fontSize: 15,
            transition: 'background 0.15s ease',
          }}
        >
          ↑
        </button>
      </form>
    </section>
  );
}
