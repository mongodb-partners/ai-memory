import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useReducer,
  useRef,
} from 'react';
import {
  browseMemories,
  fetchConfig,
  resetUser,
  streamChat,
  type MemoryEvent,
  type MemoryGroups,
  type MemoryHit,
  type ServerConfig,
} from '../api/client';

function genId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'error' | 'system';
  content: string;
  timestamp: Date;
  /** Per-turn correlation id, from the `correlation` frame. Ties the message to
   *  its `episodes` document — the thing to search for in Compass. */
  correlationId?: string;
  /** Whether memory was on for the turn that produced this message, so an
   *  OFF/ON comparison stays labelled after the toggle flips again. */
  memoryEnabled?: boolean;
}

/** What the memory panel shows for the current turn. */
export interface MemoryView {
  /** Hits recalled before generation, by tier. */
  recall: MemoryGroups;
  /** Write-back frames, in the order the server emitted them. */
  writes: MemoryEvent[];
  /** Semantic-cache outcome for this turn, or null when memory was off. */
  cache: {
    hit: boolean;
    score?: number | null;
    replayed?: boolean;
    match?: string | null;
  } | null;
  /** True while a turn is in flight and recall has not reported yet. */
  pending: boolean;
}

const EMPTY_GROUPS: MemoryGroups = { stm: [], ltm: [], episodic: [] };
const EMPTY_VIEW: MemoryView = {
  recall: EMPTY_GROUPS,
  writes: [],
  cache: null,
  pending: false,
};

interface State {
  messages: Message[];
  isLoading: boolean;
  userId: string;
  threadId: string;
  streamingId: string | null;
  memoryEnabled: boolean;
  memory: MemoryView;
  config: ServerConfig | null;
  /** Set when the panel is showing stored documents rather than a turn's
   *  recall, so the header can say which. */
  browsing: boolean;
}

type Action =
  | { type: 'SET_USER_ID'; userId: string }
  | { type: 'SET_MEMORY_ENABLED'; enabled: boolean }
  | { type: 'SET_CONFIG'; config: ServerConfig | null }
  | { type: 'ADD_MSG'; msg: Message }
  | { type: 'START_ASSISTANT'; id: string; memoryEnabled: boolean }
  | { type: 'APPEND_TOKEN'; id: string; token: string }
  | { type: 'SET_CORRELATION'; id: string; correlationId: string }
  | { type: 'MEMORY_FRAME'; ev: MemoryEvent }
  | { type: 'BROWSE'; groups: MemoryGroups }
  | { type: 'FINALIZE' }
  | { type: 'REPLACE_WITH_ERROR'; id: string; detail: string }
  | { type: 'CLEAR'; threadId: string };

/**
 * Fold one `memory` frame into the panel state.
 *
 * Recall frames replace their tier rather than appending: one turn produces
 * exactly one recall per tier, and appending would stack the previous turn's
 * hits underneath and misrepresent what this turn actually retrieved.
 */
function applyMemoryFrame(view: MemoryView, ev: MemoryEvent): MemoryView {
  if (ev.phase === 'recall') {
    const tier = ev.tier as keyof MemoryGroups;
    if (tier !== 'stm' && tier !== 'ltm' && tier !== 'episodic') return view;
    return {
      ...view,
      pending: false,
      recall: { ...view.recall, [tier]: ev.hits ?? [] },
    };
  }
  if (ev.phase === 'cache') {
    return {
      ...view,
      cache: {
        hit: Boolean(ev.cache_hit),
        // A hit emits two frames: the first carries score/match, the second only
        // reports the replay finished. Fall back to the retained value so the
        // second frame does not blank the numbers the audience is reading.
        score: ev.score ?? view.cache?.score ?? null,
        match: ev.match ?? view.cache?.match ?? null,
        replayed: ev.replayed,
      },
      // A cache hit answers without recall, so nothing further is coming.
      pending: ev.replayed ? false : view.pending,
    };
  }
  // phase === 'write'
  return { ...view, writes: [...view.writes, ev] };
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'SET_USER_ID':
      return { ...state, userId: action.userId };
    case 'SET_MEMORY_ENABLED':
      return { ...state, memoryEnabled: action.enabled };
    case 'SET_CONFIG':
      return { ...state, config: action.config };
    case 'ADD_MSG':
      return { ...state, messages: [...state.messages, action.msg] };
    case 'START_ASSISTANT':
      return {
        ...state,
        isLoading: true,
        streamingId: action.id,
        browsing: false,
        // Reset the panel at the start of every turn. Carrying the previous
        // turn's hits over would let a memory-OFF turn display the memory-ON
        // turn's recall — the exact false impression the demo must not give.
        memory: action.memoryEnabled
          ? { ...EMPTY_VIEW, pending: true }
          : EMPTY_VIEW,
        messages: [
          ...state.messages,
          {
            id: action.id,
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            memoryEnabled: action.memoryEnabled,
          },
        ],
      };
    case 'APPEND_TOKEN':
      return {
        ...state,
        memory: { ...state.memory, pending: false },
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, content: m.content + action.token } : m,
        ),
      };
    case 'SET_CORRELATION':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, correlationId: action.correlationId } : m,
        ),
      };
    case 'MEMORY_FRAME':
      return { ...state, memory: applyMemoryFrame(state.memory, action.ev) };
    case 'BROWSE':
      return {
        ...state,
        browsing: true,
        memory: { ...EMPTY_VIEW, recall: action.groups },
      };
    case 'FINALIZE':
      return {
        ...state,
        isLoading: false,
        streamingId: null,
        memory: { ...state.memory, pending: false },
      };
    case 'REPLACE_WITH_ERROR':
      return {
        ...state,
        isLoading: false,
        streamingId: null,
        memory: { ...state.memory, pending: false },
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, role: 'error', content: action.detail } : m,
        ),
      };
    case 'CLEAR':
      return {
        ...state,
        messages: [],
        isLoading: false,
        streamingId: null,
        threadId: action.threadId,
        browsing: false,
        memory: EMPTY_VIEW,
      };
    default:
      return state;
  }
}

interface ChatContextValue {
  messages: Message[];
  isLoading: boolean;
  userId: string;
  threadId: string;
  memoryEnabled: boolean;
  memory: MemoryView;
  config: ServerConfig | null;
  browsing: boolean;
  setUserId: (u: string) => void;
  setMemoryEnabled: (enabled: boolean) => void;
  sendMessage: (message: string, opts?: { newThread?: boolean }) => void;
  newConversation: () => void;
  /** Load stored memories into the panel without spending a turn. */
  refreshMemories: () => void;
  /** Wipe this user's memories. Destructive; the caller confirms. */
  wipeUser: () => Promise<void>;
}

const ChatContext = createContext<ChatContextValue | null>(null);

const initialState: State = {
  messages: [],
  isLoading: false,
  userId: 'ai4-demo',
  threadId: genId(),
  streamingId: null,
  // Starts ON. The demo's arc is ON → OFF → ON, and a UI that boots into the
  // broken case invites the audience to judge the wrong thing first.
  memoryEnabled: true,
  memory: EMPTY_VIEW,
  config: null,
  browsing: false,
};

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const loadingRef = useRef(false);
  // Holds the in-flight stream's controller. Without this, switching user or
  // starting a new thread leaves the old fetch running and its late onToken
  // callbacks bleed the previous answer into the new conversation.
  const streamRef = useRef<AbortController | null>(null);
  // Cancels an in-flight browse so it cannot overwrite a turn's live recall.
  const browseRef = useRef<{ cancelled: boolean } | null>(null);

  useEffect(() => () => streamRef.current?.abort(), []);

  // The header shows the model the server is actually running. Read once from
  // /config rather than hardcoded: a slide claiming one model while the server
  // runs another is the kind of error an audience catches.
  useEffect(() => {
    let live = true;
    fetchConfig().then((config) => {
      if (live) dispatch({ type: 'SET_CONFIG', config });
    });
    return () => {
      live = false;
    };
  }, []);

  // Switching user is the per-user-isolation proof, so the panel must follow
  // the user. Debounced because the Header writes userId on every keystroke.
  useEffect(() => {
    const userId = state.userId;
    if (!userId) return;
    const token = { cancelled: false };
    browseRef.current = token;
    const timer = setTimeout(async () => {
      const groups = await browseMemories(userId);
      if (token.cancelled) return;
      // A deliberate context switch — drop any live stream.
      streamRef.current?.abort();
      streamRef.current = null;
      loadingRef.current = false;
      dispatch({ type: 'CLEAR', threadId: genId() });
      dispatch({ type: 'BROWSE', groups });
    }, 450);
    return () => {
      token.cancelled = true;
      clearTimeout(timer);
    };
  }, [state.userId]);

  const setUserId = useCallback((u: string) => {
    dispatch({ type: 'SET_USER_ID', userId: u });
  }, []);

  const setMemoryEnabled = useCallback((enabled: boolean) => {
    dispatch({ type: 'SET_MEMORY_ENABLED', enabled });
  }, []);

  const newConversation = useCallback(() => {
    if (browseRef.current) browseRef.current.cancelled = true;
    streamRef.current?.abort();
    streamRef.current = null;
    loadingRef.current = false;
    dispatch({ type: 'CLEAR', threadId: genId() });
  }, []);

  const refreshMemories = useCallback(() => {
    const userId = state.userId;
    if (!userId) return;
    const token = { cancelled: false };
    browseRef.current = token;
    browseMemories(userId).then((groups) => {
      if (token.cancelled) return;
      dispatch({ type: 'BROWSE', groups });
    });
  }, [state.userId]);

  const wipeUser = useCallback(async () => {
    const userId = state.userId;
    if (!userId) return;
    streamRef.current?.abort();
    streamRef.current = null;
    loadingRef.current = false;
    const err = await resetUser(userId);
    dispatch({ type: 'CLEAR', threadId: genId() });
    if (err) {
      dispatch({
        type: 'ADD_MSG',
        msg: {
          id: genId(),
          role: 'system',
          content: `Reset failed: ${err}`,
          timestamp: new Date(),
        },
      });
      return;
    }
    dispatch({
      type: 'ADD_MSG',
      msg: {
        id: genId(),
        role: 'system',
        content: `Wiped every memory for “${userId}”. The next turn starts from nothing.`,
        timestamp: new Date(),
      },
    });
  }, [state.userId]);

  const sendMessage = useCallback(
    (message: string, opts?: { newThread?: boolean }) => {
      const trimmed = message.trim();
      if (!trimmed || loadingRef.current || !state.userId) return;

      // A browse in flight would land after this turn's recall and overwrite it.
      if (browseRef.current) browseRef.current.cancelled = true;
      loadingRef.current = true;

      streamRef.current?.abort();
      const ctrl = new AbortController();
      streamRef.current = ctrl;

      // `newThread` mints a fresh thread id *atomically* and uses the local
      // value for the request — `state.threadId` is a stale closure until the
      // CLEAR re-renders. A fresh thread is how the demo proves recall crosses
      // conversations rather than just reading back the transcript.
      const threadId = opts?.newThread ? genId() : state.threadId;
      if (opts?.newThread) dispatch({ type: 'CLEAR', threadId });

      const memoryEnabled = state.memoryEnabled;
      dispatch({
        type: 'ADD_MSG',
        msg: {
          id: genId(),
          role: 'user',
          content: trimmed,
          timestamp: new Date(),
          memoryEnabled,
        },
      });

      const assistantId = genId();
      dispatch({ type: 'START_ASSISTANT', id: assistantId, memoryEnabled });

      streamChat(
        {
          user_id: state.userId,
          thread_id: threadId,
          message: trimmed,
          memory_enabled: memoryEnabled,
        },
        {
          onToken: (token) => {
            if (ctrl.signal.aborted) return;
            dispatch({ type: 'APPEND_TOKEN', id: assistantId, token });
          },
          onCorrelation: (correlationId) => {
            if (ctrl.signal.aborted) return;
            dispatch({ type: 'SET_CORRELATION', id: assistantId, correlationId });
          },
          onMemory: (ev) => {
            if (ctrl.signal.aborted) return;
            dispatch({ type: 'MEMORY_FRAME', ev });
          },
          onDone: () => {
            if (ctrl.signal.aborted) return;
            dispatch({ type: 'FINALIZE' });
            loadingRef.current = false;
          },
          onError: (detail) => {
            if (ctrl.signal.aborted) return;
            dispatch({ type: 'REPLACE_WITH_ERROR', id: assistantId, detail });
            loadingRef.current = false;
          },
        },
        ctrl.signal,
      );
    },
    [state.userId, state.threadId, state.memoryEnabled],
  );

  return (
    <ChatContext.Provider
      value={{
        messages: state.messages,
        isLoading: state.isLoading,
        userId: state.userId,
        threadId: state.threadId,
        memoryEnabled: state.memoryEnabled,
        memory: state.memory,
        config: state.config,
        browsing: state.browsing,
        setUserId,
        setMemoryEnabled,
        sendMessage,
        newConversation,
        refreshMemories,
        wipeUser,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChat must be used within ChatProvider');
  return ctx;
}

export type { MemoryHit };
