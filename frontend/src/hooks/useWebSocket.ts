import { useEffect, useRef, useState, useCallback } from 'react';
import type { PipelineStatus } from '../types';

/** WebSocket connection states */
type WsState = 'connecting' | 'connected' | 'disconnected' | 'error';

/** Return value of the useWebSocket hook */
interface UseWebSocketReturn {
  /** Latest pipeline status message, or null if none received */
  status: PipelineStatus | null;
  /** Whether the WebSocket is currently connected */
  isConnected: boolean;
  /** Last error message, or null */
  error: string | null;
  /** Manually disconnect the WebSocket */
  disconnect: () => void;
}

const WS_BASE_URL = 'ws://localhost:8000';
const MAX_RETRIES = 6;
const INITIAL_BACKOFF_MS = 1_000;

/**
 * Custom hook that maintains a WebSocket connection to the pipeline status endpoint.
 *
 * Connects to ws://localhost:8000/ws/pipeline/{spillId} and streams real-time
 * PipelineStatus JSON messages. Implements exponential backoff reconnection
 * (1s, 2s, 4s, 8s, 16s, 32s) up to MAX_RETRIES attempts.
 *
 * @param spillId - UUID of the active spill being processed. Pass empty string to disable.
 * @returns WebSocket state including latest status, connection flag, and error info.
 */
export function useWebSocket(spillId: string | null): UseWebSocketReturn {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [wsState, setWsState] = useState<WsState>('disconnected');
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shouldConnectRef = useRef(true);

  const clearRetryTimer = () => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  };

  const connect = useCallback(() => {
    if (!spillId || !shouldConnectRef.current) return;

    // Clean up any existing connection
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
    }

    const url = `${WS_BASE_URL}/ws/pipeline/${spillId}`;
    console.log(`[WS] Connecting to ${url} (attempt ${retryCountRef.current + 1})`);

    setWsState('connecting');
    setError(null);

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      setError('Failed to create WebSocket connection');
      setWsState('error');
      return;
    }

    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected');
      setWsState('connected');
      setError(null);
      retryCountRef.current = 0;
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as PipelineStatus;
        setStatus(data);

        // Stop reconnecting if pipeline completed or errored
        if (data.stage === 'complete' || data.stage === 'error') {
          shouldConnectRef.current = false;
        }
      } catch (parseError) {
        console.warn('[WS] Failed to parse message:', event.data);
      }
    };

    ws.onerror = (event) => {
      console.error('[WS] Error:', event);
      setError('WebSocket connection error');
      setWsState('error');
    };

    ws.onclose = (event) => {
      console.log(`[WS] Closed (code=${event.code}, reason=${event.reason})`);
      setWsState('disconnected');

      // Attempt reconnect with exponential backoff
      if (
        shouldConnectRef.current &&
        retryCountRef.current < MAX_RETRIES
      ) {
        const backoffMs = INITIAL_BACKOFF_MS * Math.pow(2, retryCountRef.current);
        retryCountRef.current += 1;
        console.log(`[WS] Reconnecting in ${backoffMs}ms (retry ${retryCountRef.current}/${MAX_RETRIES})`);

        retryTimerRef.current = setTimeout(() => {
          connect();
        }, backoffMs);
      } else if (retryCountRef.current >= MAX_RETRIES) {
        setError(`Failed to connect after ${MAX_RETRIES} attempts`);
        setWsState('error');
      }
    };
  }, [spillId]);

  // Connect / disconnect when spillId changes
  useEffect(() => {
    if (!spillId) {
      setStatus(null);
      setWsState('disconnected');
      setError(null);
      return;
    }

    shouldConnectRef.current = true;
    retryCountRef.current = 0;
    connect();

    return () => {
      // Cleanup on unmount or spillId change
      shouldConnectRef.current = false;
      clearRetryTimer();
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
      setStatus(null);
      setWsState('disconnected');
    };
  }, [spillId, connect]);

  const disconnect = useCallback(() => {
    shouldConnectRef.current = false;
    clearRetryTimer();
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    setWsState('disconnected');
  }, []);

  return {
    status,
    isConnected: wsState === 'connected',
    error,
    disconnect,
  };
}
