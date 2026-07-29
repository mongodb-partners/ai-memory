import React from 'react';
import ChatPanel from './components/ChatPanel';
import Header from './components/Header';
import MemorySection from './components/MemorySection';
import { ChatProvider } from './context/ChatContext';

/**
 * One screen: chat on the left, memory on the right.
 *
 * No routes, no landing page, no hero. The retail app this borrows from had 576
 * lines of decorative WebGL above the fold; at a booth the audience gets fifteen
 * minutes and every pixel has to be doing work. The two panels are side by side
 * and always visible because the demo's claim is a *correspondence* — this
 * answer came from those documents — and a correspondence you have to scroll to
 * see is one the audience will not follow.
 */

const shellStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  height: '100vh',
  overflow: 'hidden',
};

const splitStyle: React.CSSProperties = {
  flex: 1,
  display: 'grid',
  // Chat gets the larger share; the memory panel needs enough width for a hit's
  // text plus its score row on one line.
  gridTemplateColumns: 'minmax(0, 1.35fr) minmax(0, 1fr)',
  gap: 14,
  padding: 14,
  minHeight: 0,
  overflow: 'hidden',
};

export default function App() {
  return (
    <ChatProvider>
      <div style={shellStyle}>
        <Header />
        <main style={splitStyle} className="split">
          <ChatPanel />
          <MemorySection />
        </main>
      </div>
    </ChatProvider>
  );
}
