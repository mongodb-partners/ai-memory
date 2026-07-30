import React, { useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message } from '../context/ChatContext';

interface Props {
  message: Message;
}

/**
 * Guard against an UNCLOSED ``` fence in the model's text. A model asked about
 * MongoDB will happily open a fence around an aggregation pipeline and forget to
 * close it, which makes react-markdown render everything after it — headings,
 * tables, prose — as one raw code block. If the fence count is odd, drop the
 * last unmatched marker.
 */
function balanceCodeFences(md: string): string {
  const fences = md.match(/```/g);
  if (!fences || fences.length % 2 === 0) return md;
  const i = md.lastIndexOf('```');
  return md.slice(0, i) + md.slice(i + 3);
}

const userBubbleStyle: React.CSSProperties = {
  maxWidth: '86%',
  padding: '9px 12px',
  borderRadius: 'var(--radius-md) var(--radius-md) 4px var(--radius-md)',
  background: 'var(--green-tint-12)',
  border: '1px solid var(--green-border)',
  color: 'var(--text)',
  fontSize: 14,
  lineHeight: 1.55,
  alignSelf: 'flex-end',
  wordBreak: 'break-word',
  whiteSpace: 'pre-wrap',
};

const assistantBubbleStyle: React.CSSProperties = {
  maxWidth: '92%',
  padding: '10px 14px',
  borderRadius: 'var(--radius-md) var(--radius-md) var(--radius-md) 4px',
  background: 'rgba(255,255,255,0.05)',
  border: '1px solid var(--border)',
  color: 'var(--text)',
  fontSize: 14,
  lineHeight: 1.7,
  alignSelf: 'flex-start',
  wordBreak: 'break-word',
};

/** Memory-off answers get a muted, dashed bubble. The whole demo is a
 *  comparison, and once the transcript scrolls the audience needs to tell the
 *  two halves apart without remembering which way the toggle was set. */
const noMemoryBubbleStyle: React.CSSProperties = {
  ...assistantBubbleStyle,
  background: 'rgba(255,255,255,0.02)',
  border: '1px dashed rgba(123,140,154,0.45)',
  color: 'var(--text-secondary)',
};

const errorBubbleStyle: React.CSSProperties = {
  ...assistantBubbleStyle,
  background: 'var(--danger-tint)',
  border: '1px solid rgba(239,68,68,0.4)',
  color: 'var(--danger)',
};

const roleTagStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '1.5px',
  marginBottom: 5,
  color: 'var(--text-secondary)',
  display: 'flex',
  alignItems: 'center',
  gap: 8,
};

const offTagStyle: React.CSSProperties = {
  color: 'var(--warning, #ffc010)',
  letterSpacing: '1px',
};

const systemNoteStyle: React.CSSProperties = {
  alignSelf: 'center',
  fontSize: 12,
  color: 'var(--text-secondary)',
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-pill)',
  padding: '6px 12px',
  margin: '2px 0',
  maxWidth: '85%',
  textAlign: 'center',
  lineHeight: 1.5,
};

const correlationStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  color: 'var(--text-secondary)',
  opacity: 0.75,
  marginTop: 5,
  paddingLeft: 2,
  userSelect: 'all',
};

const ChatMessage: React.FC<Props> = React.memo(({ message }) => {
  if (message.role === 'system') {
    return <div style={systemNoteStyle}>{message.content}</div>;
  }

  const isUser = message.role === 'user';
  const isError = message.role === 'error';
  const memoryOff = message.memoryEnabled === false;

  const bubbleStyle = isError
    ? errorBubbleStyle
    : isUser
      ? userBubbleStyle
      : memoryOff
        ? noMemoryBubbleStyle
        : assistantBubbleStyle;

  // Keep code rendering plain — no rehype-raw, no dangerouslySetInnerHTML.
  const renderCode = useCallback(
    (props: { className?: string; children?: React.ReactNode }) => {
      const { className, children } = props;
      return <code className={className}>{children}</code>;
    },
    [],
  );

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      {!isUser && (
        <div style={roleTagStyle}>
          <span>{isError ? 'Error' : 'Assistant'}</span>
          {memoryOff && !isError && <span style={offTagStyle}>memory off</span>}
        </div>
      )}
      <div style={bubbleStyle}>
        {isUser ? (
          <div>{message.content}</div>
        ) : (
          <div className="chat-markdown">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{ code: renderCode }}
            >
              {balanceCodeFences(message.content || (isError ? '' : '…'))}
            </ReactMarkdown>
          </div>
        )}
      </div>
      {/* The correlation id is the join key to this turn's `episodes` document.
          Showing it means the Compass screen is a lookup of something the
          audience just watched happen, not a document picked in advance. */}
      {!isUser && !isError && message.correlationId && (
        <div style={correlationStyle}>correlation_id: {message.correlationId}</div>
      )}
    </div>
  );
});

ChatMessage.displayName = 'ChatMessage';
export default ChatMessage;
