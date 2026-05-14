import { useCallback, useEffect, useRef, useState } from 'react';
import * as api from '../../services/api';
import Button from '../ui/Button';

type Status = api.VectorstoreStatus;
type QueryResult = api.TestQueryResult;

const EMBEDDING_MODELS = [
  { id: 'text-embedding-3-small', label: 'text-embedding-3-small (recommended, cheap)' },
  { id: 'text-embedding-3-large', label: 'text-embedding-3-large (highest quality, ~6× cost)' },
  { id: 'text-embedding-ada-002', label: 'text-embedding-ada-002 (legacy)' },
];

const TAG_OPTIONS = ['sales', 'negotiation'];

export default function VectorstorePanel() {
  const [status, setStatus] = useState<Status | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(true);

  // Rebuild form
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [embeddingModel, setEmbeddingModel] = useState('text-embedding-3-small');
  const [tagsFilter, setTagsFilter] = useState<string[]>([]);

  // Rebuild job
  const [job, setJob] = useState<api.RebuildJob | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Test query
  const [query, setQuery] = useState('');
  const [queryK, setQueryK] = useState(5);
  const [queryTarget, setQueryTarget] = useState<'live' | 'staging'>('live');
  const [queryTagsFilter, setQueryTagsFilter] = useState<string[]>([]);
  const [queryResults, setQueryResults] = useState<QueryResult[]>([]);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);

  const [error, setError] = useState<string | null>(null);

  const loadStatus = useCallback(async () => {
    try {
      const s = await api.adminGetVectorstoreStatus();
      setStatus(s);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load vectorstore status');
    } finally {
      setLoadingStatus(false);
    }
  }, []);

  useEffect(() => { loadStatus(); }, [loadStatus]);

  // Clean up polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const startPolling = (id: string) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const j = await api.adminGetRebuildJob(id);
        setJob(j);
        if (j.status === 'done' || j.status === 'error') {
          stopPolling();
          loadStatus();
        }
      } catch {
        stopPolling();
      }
    }, 2000);
  };

  const handleRebuild = async () => {
    setError(null);
    try {
      const { job_id } = await api.adminTriggerRebuild({
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        embedding_model: embeddingModel,
        tags_filter: tagsFilter.length ? tagsFilter : undefined,
      });
      setJob({ id: job_id, status: 'pending', percent: 0, current_file: null, errors: [], params: {}, started_at: new Date().toISOString(), finished_at: null });
      startPolling(job_id);
    } catch (e: any) {
      setError(e.response?.data?.detail ?? 'Rebuild failed to start');
    }
  };

  const handleTestQuery = async () => {
    if (!query.trim()) return;
    setQueryLoading(true);
    setQueryError(null);
    setQueryResults([]);
    try {
      const results = await api.adminTestQuery({
        query: query.trim(),
        k: queryK,
        target: queryTarget,
        tags_filter: queryTagsFilter.length ? queryTagsFilter : undefined,
      });
      setQueryResults(results);
    } catch (e: any) {
      setQueryError(e.response?.data?.detail ?? 'Query failed');
    } finally {
      setQueryLoading(false);
    }
  };

  const toggleTagFilter = (tag: string, setter: React.Dispatch<React.SetStateAction<string[]>>) => {
    setter(prev => prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]);
  };

  const isRunning = job?.status === 'running' || job?.status === 'pending';

  return (
    <div className="space-y-8">
      {error && (
        <div className="px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-sm">
          ⚠️ {error}
          <button className="ml-3 underline text-xs" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      {/* Status card */}
      <div className="bg-chat-muted/40 rounded-lg p-5 border border-chat-border">
        <h3 className="font-semibold text-chat-foreground mb-3">Current Vectorstore</h3>
        {loadingStatus ? (
          <p className="text-chat-muted-foreground text-sm">Loading…</p>
        ) : status ? (
          <dl className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
            <Stat label="Index exists" value={status.index_exists ? '✓ Yes' : '✗ No'} />
            <Stat label="Embedding model" value={status.embedding_model ?? '—'} />
            <Stat label="Chunks" value={status.chunks_count?.toLocaleString() ?? '—'} />
            <Stat label="Dimensions" value={status.dimensions?.toString() ?? '—'} />
            <Stat label="Last build" value={status.last_build ? new Date(status.last_build).toLocaleString() : '—'} />
            <Stat label="Size" value={status.size_bytes ? `${(status.size_bytes / 1024 / 1024).toFixed(1)} MB` : '—'} />
          </dl>
        ) : (
          <p className="text-chat-muted-foreground text-sm">No status available.</p>
        )}
      </div>

      {/* Rebuild form */}
      <div className="space-y-4">
        <h3 className="font-semibold text-chat-foreground">Rebuild Vectorstore</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className="space-y-1">
            <span className="text-sm text-chat-muted-foreground">Chunk size (chars)</span>
            <input
              type="number"
              value={chunkSize}
              min={200}
              max={4000}
              onChange={e => setChunkSize(parseInt(e.target.value) || 1000)}
              className="w-full bg-chat-muted border border-chat-border rounded px-3 py-2 text-sm text-chat-foreground"
            />
          </label>
          <label className="space-y-1">
            <span className="text-sm text-chat-muted-foreground">Chunk overlap (chars)</span>
            <input
              type="number"
              value={chunkOverlap}
              min={0}
              max={1000}
              onChange={e => setChunkOverlap(parseInt(e.target.value) || 200)}
              className="w-full bg-chat-muted border border-chat-border rounded px-3 py-2 text-sm text-chat-foreground"
            />
          </label>
        </div>

        <label className="block space-y-1">
          <span className="text-sm text-chat-muted-foreground">Embedding model</span>
          <select
            value={embeddingModel}
            onChange={e => setEmbeddingModel(e.target.value)}
            className="w-full bg-chat-muted border border-chat-border rounded px-3 py-2 text-sm text-chat-foreground"
          >
            {EMBEDDING_MODELS.map(m => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </label>

        <div className="space-y-1">
          <span className="text-sm text-chat-muted-foreground">Restrict to tags (leave empty for all sources)</span>
          <div className="flex gap-2 mt-1">
            {TAG_OPTIONS.map(tag => (
              <button
                key={tag}
                onClick={() => toggleTagFilter(tag, setTagsFilter)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  tagsFilter.includes(tag)
                    ? 'bg-chat-primary text-white'
                    : 'bg-chat-muted text-chat-muted-foreground border border-chat-border hover:bg-chat-primary/20'
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>

        <Button onClick={handleRebuild} disabled={isRunning}>
          {isRunning ? 'Rebuilding…' : 'Start rebuild'}
        </Button>
      </div>

      {/* Job progress */}
      {job && (
        <div className="bg-chat-muted/40 rounded-lg p-5 border border-chat-border space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-chat-foreground">Rebuild job</h3>
            <span className={`text-xs font-medium px-2 py-0.5 rounded ${
              job.status === 'done' ? 'bg-success/20 text-success' :
              job.status === 'error' ? 'bg-danger/20 text-danger' :
              'bg-chat-primary/20 text-chat-primary'
            }`}>
              {job.status}
            </span>
          </div>

          {/* Progress bar */}
          <div className="h-2 bg-chat-muted rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 rounded-full ${
                job.status === 'error' ? 'bg-danger' : 'bg-chat-primary'
              }`}
              style={{ width: `${job.percent}%` }}
            />
          </div>
          <p className="text-xs text-chat-muted-foreground">
            {job.percent}% {job.current_file ? `— ${job.current_file}` : ''}
          </p>

          {job.errors.length > 0 && (
            <div className="text-xs text-danger space-y-1">
              {job.errors.map((e, i) => <p key={i}>⚠️ {e}</p>)}
            </div>
          )}
        </div>
      )}

      {/* Test query */}
      <div className="space-y-4">
        <h3 className="font-semibold text-chat-foreground">Test query</h3>

        <div className="flex gap-3">
          <input
            className="flex-1 bg-chat-muted border border-chat-border rounded px-3 py-2 text-sm text-chat-foreground placeholder:text-chat-muted-foreground"
            placeholder="e.g. how do I anchor a price in negotiation?"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleTestQuery()}
          />
          <Button onClick={handleTestQuery} disabled={queryLoading || !query.trim()}>
            {queryLoading ? 'Searching…' : 'Search'}
          </Button>
        </div>

        <div className="flex gap-4 text-sm">
          <label className="flex items-center gap-2 text-chat-muted-foreground">
            Target:
            <select
              value={queryTarget}
              onChange={e => setQueryTarget(e.target.value as 'live' | 'staging')}
              className="bg-chat-muted border border-chat-border rounded px-2 py-1 text-chat-foreground text-sm"
            >
              <option value="live">Live</option>
              <option value="staging">Staging</option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-chat-muted-foreground">
            Top k:
            <input
              type="number"
              value={queryK}
              min={1}
              max={20}
              onChange={e => setQueryK(parseInt(e.target.value) || 5)}
              className="w-16 bg-chat-muted border border-chat-border rounded px-2 py-1 text-chat-foreground text-sm"
            />
          </label>
          <div className="flex items-center gap-2 text-chat-muted-foreground">
            Tags:
            {TAG_OPTIONS.map(tag => (
              <button
                key={tag}
                onClick={() => toggleTagFilter(tag, setQueryTagsFilter)}
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  queryTagsFilter.includes(tag)
                    ? 'bg-chat-primary text-white'
                    : 'bg-chat-muted border border-chat-border text-chat-muted-foreground hover:bg-chat-primary/20'
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>

        {queryError && (
          <p className="text-danger text-sm">⚠️ {queryError}</p>
        )}

        {queryResults.length > 0 && (
          <div className="space-y-3">
            {queryResults.map((r, i) => (
              <div key={i} className="bg-chat-muted/40 border border-chat-border rounded-lg p-4 text-sm">
                <div className="flex gap-2 mb-2 flex-wrap">
                  <span className="font-mono text-xs text-chat-muted-foreground">{r.source_file}</span>
                  {r.page != null && <span className="text-xs text-chat-muted-foreground">p.{r.page}</span>}
                  <span className="text-xs text-chat-muted-foreground">score: {r.score.toFixed(4)}</span>
                  {r.tags.map(t => (
                    <span key={t} className="px-1.5 py-0.5 rounded text-xs bg-chat-primary/20 text-chat-primary">{t}</span>
                  ))}
                </div>
                <p className="text-chat-foreground text-xs leading-relaxed whitespace-pre-wrap">{r.content}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-chat-muted-foreground text-xs">{label}</dt>
      <dd className="text-chat-foreground font-medium">{value}</dd>
    </div>
  );
}
