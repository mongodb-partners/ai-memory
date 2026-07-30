/**
 * The demo backend's HTTP surface. Five routes, one of them streaming.
 *
 * The SSE consumer below is a general-purpose one: `correlation` first, then any
 * number of `token` and `memory` frames, then a terminal `done` or `error`.
 * `memory` is the frame that matters — a chat UI that streams tokens beautifully
 * and says nothing about what the agent remembered is the usual case, and it is
 * exactly the gap this demo exists to close.
 */

// Always relative. Vite proxies /api → the demo server in dev; a reverse proxy
// does the same in prod, so one code path serves both.
const API_BASE = '/api';

export type Tier = 'stm' | 'ltm' | 'episodic' | 'cache';
export type MemoryPhase = 'recall' | 'write' | 'cache';

/** One recalled or written item, already projected by the server. */
export interface MemoryHit {
  text: string;
  /**
   * The raw fused score. For recalled `memories` this is the calibrated
   * `final_score`; for `episodes` it is the `$rankFusion` RRF sum, which is
   * ~1/61 for a first-place hit — see `rank` for the readable form.
   */
  score?: number | null;
  /** Server-rendered label, e.g. `#1 · rrf 0.0164`. Display this, not `score`. */
  rank?: string;
  importance?: number | null;
  access_count?: number | null;
  ts?: string | null;
  tier?: string | null;
  step?: number | null;
  tools?: string[];
  files?: string[];
}

export interface MemoryEvent {
  phase: MemoryPhase;
  tier: Tier;
  query?: string;
  hits: MemoryHit[];
  cache_hit?: boolean;
  replayed?: boolean;
  score?: number | null;
  /** `'exact'` or `'semantic'` on a cache hit — which path matched. */
  match?: string | null;
  note?: string;
  error?: string;
  correlation_id?: string;
  elapsed_ms?: number;
}

export interface MemoryGroups {
  stm: MemoryHit[];
  ltm: MemoryHit[];
  episodic: MemoryHit[];
}

export interface BrowseResponse {
  user_id: string;
  query: string;
  groups: MemoryGroups;
}

export interface ServerConfig {
  llm_provider: string;
  llm_model: string;
  embedding_provider: string;
  embedding_model: string;
  embedding_dimension: number;
  database: string;
}

export interface ChatRequest {
  user_id: string;
  thread_id: string;
  message: string;
  memory_enabled: boolean;
}

export interface StreamHandlers {
  onToken: (token: string) => void;
  /** First frame of every stream: the turn's correlation id. */
  onCorrelation?: (correlationId: string) => void;
  /** Non-terminal. Fires once per tier per phase, several times a turn. */
  onMemory?: (ev: MemoryEvent) => void;
  onDone: () => void;
  onError: (detail: string) => void;
}

/** GET /api/config — the model actually in use, for the header. */
export async function fetchConfig(): Promise<ServerConfig | null> {
  try {
    const res = await fetch(`${API_BASE}/config`);
    if (!res.ok) return null;
    return (await res.json()) as ServerConfig;
  } catch {
    return null;
  }
}

/**
 * GET /api/memories — browse a user's memories without spending a turn.
 *
 * An empty `query` lists the most recent documents; a non-empty one runs the
 * same hybrid search a turn would.
 */
export async function browseMemories(
  userId: string,
  query = '',
  limit = 20,
): Promise<MemoryGroups> {
  const empty: MemoryGroups = { stm: [], ltm: [], episodic: [] };
  try {
    const params = new URLSearchParams({
      user_id: userId,
      query,
      limit: String(limit),
    });
    const res = await fetch(`${API_BASE}/memories?${params.toString()}`);
    if (!res.ok) return empty;
    const body = (await res.json()) as BrowseResponse;
    return {
      stm: body.groups?.stm ?? [],
      ltm: body.groups?.ltm ?? [],
      episodic: body.groups?.episodic ?? [],
    };
  } catch {
    return empty;
  }
}

/**
 * POST /api/reset — wipe a demo user. `confirm` is required server-side.
 *
 * Resolves to an error string rather than throwing, so a failed reset shows in
 * the UI instead of as an unhandled rejection during a rehearsal.
 */
export async function resetUser(userId: string): Promise<string | null> {
  try {
    const res = await fetch(`${API_BASE}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, confirm: true }),
    });
    if (!res.ok) return `HTTP ${res.status}`;
    return null;
  } catch (err) {
    return err instanceof Error ? err.message : 'network error';
  }
}

/** POST /api/chat as a streaming fetch. */
export async function streamChat(
  req: ChatRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(req),
      signal,
    });
  } catch (err) {
    // A deliberate abort (new thread / unmount) is not an error.
    if (err instanceof DOMException && err.name === 'AbortError') return;
    handlers.onError(err instanceof Error ? err.message : 'network error');
    return;
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body && body.detail) detail = String(body.detail);
    } catch {
      /* non-JSON error body; keep the status */
    }
    handlers.onError(detail);
    return;
  }

  if (!res.body) {
    handlers.onError('empty response body');
    return;
  }

  await consumeSSEStream(res, handlers);
}

/**
 * Read `res.body`, split it into SSE frames, dispatch each `event:`.
 *
 * Two details that are easy to get wrong and both matter here:
 *
 * 1. **Whitespace.** A frame is `data: ` + value, and a conforming client strips
 *    exactly ONE leading space. Streamed tokens carry their own leading space
 *    (`" the"`), so the wire shows `data:  the` — two spaces. Calling `.trim()`
 *    on the data line, which looks obviously right, eats every word boundary and
 *    renders the answer as "Roastthepeppers". Hence `.replace(/^ /, '')`.
 * 2. **Terminal frames.** If the stream ends without `done` or `error`, the
 *    connection was truncated. Report that rather than a phantom success.
 */
async function consumeSSEStream(
  res: Response,
  handlers: StreamHandlers,
): Promise<void> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let sawTerminal = false;

  while (true) {
    let value: Uint8Array | undefined;
    let done: boolean;
    try {
      ({ value, done } = await reader.read());
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      handlers.onError(err instanceof Error ? err.message : 'stream read error');
      return;
    }
    if (done) break;
    // sse_starlette terminates frames with CRLF CRLF; the spec also allows
    // LF LF. Normalize up front so the split is one search for "\n\n".
    buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, '\n');

    let idx: number;
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let eventName = 'message';
      const dataLines: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
      }
      const data = dataLines.join('\n');

      if (eventName === 'token') {
        handlers.onToken(data);
      } else if (eventName === 'correlation') {
        handlers.onCorrelation?.(data);
      } else if (eventName === 'memory') {
        if (handlers.onMemory) {
          try {
            handlers.onMemory(JSON.parse(data) as MemoryEvent);
          } catch {
            /* malformed frame; a dropped panel row must not kill the answer */
          }
        }
      } else if (eventName === 'done') {
        sawTerminal = true;
        handlers.onDone();
        return;
      } else if (eventName === 'error') {
        sawTerminal = true;
        handlers.onError(data);
        return;
      }
      // Anything else (sse_starlette keepalives) is ignored by design.
    }
  }

  if (!sawTerminal) {
    handlers.onError('stream ended unexpectedly (no terminal frame)');
  }
}
