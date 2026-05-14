import { useEffect, useState } from 'react';
import * as api from '../../services/api';
import Button from '../ui/Button';
import SourceUploadDropzone from './SourceUploadDropzone';

type Source = api.AdminSource;

const TAG_OPTIONS = ['sales', 'negotiation'];

interface RowEdits {
  title: string;
  author: string;
  year: string;
  tags: string[];
  enabled: boolean;
}

export default function SourcesTab() {
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editingFilename, setEditingFilename] = useState<string | null>(null);
  const [edits, setEdits] = useState<RowEdits>({ title: '', author: '', year: '', tags: [], enabled: true });
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const load = async () => {
    try {
      const data = await api.adminListSources();
      setSources(data);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load sources');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleUpload = async (files: File[]) => {
    setUploading(true);
    setError(null);
    for (const file of files) {
      try {
        await api.adminUploadSource(file);
      } catch (e: any) {
        setError(e.response?.data?.detail ?? e.message ?? `Upload failed: ${file.name}`);
      }
    }
    setUploading(false);
    await load();
  };

  const startEdit = (s: Source) => {
    setEditingFilename(s.filename);
    setEdits({
      title: s.title ?? '',
      author: s.author ?? '',
      year: s.year?.toString() ?? '',
      tags: s.tags,
      enabled: s.enabled,
    });
  };

  const saveEdit = async (filename: string) => {
    try {
      await api.adminUpdateSource(filename, {
        title: edits.title || undefined,
        author: edits.author || undefined,
        year: edits.year ? parseInt(edits.year) : undefined,
        tags: edits.tags,
        enabled: edits.enabled,
      });
      setEditingFilename(null);
      await load();
    } catch (e: any) {
      setError(e.response?.data?.detail ?? 'Update failed');
    }
  };

  const handleDelete = async (filename: string) => {
    try {
      await api.adminDeleteSource(filename);
      setConfirmDelete(null);
      await load();
    } catch (e: any) {
      setError(e.response?.data?.detail ?? 'Delete failed');
    }
  };

  const toggleTag = (tag: string) => {
    setEdits(prev => ({
      ...prev,
      tags: prev.tags.includes(tag) ? prev.tags.filter(t => t !== tag) : [...prev.tags, tag],
    }));
  };

  if (loading) return <div className="p-4 text-chat-muted-foreground">Loading sources…</div>;

  return (
    <div className="space-y-6">
      {error && (
        <div className="px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-sm">
          ⚠️ {error}
          <button className="ml-3 underline text-xs" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <SourceUploadDropzone onUpload={handleUpload} uploading={uploading} />

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left border-separate border-spacing-0">
          <thead>
            <tr className="text-chat-muted-foreground border-b border-chat-border">
              <th className="pb-2 pr-4 font-medium">File</th>
              <th className="pb-2 pr-4 font-medium">Title</th>
              <th className="pb-2 pr-4 font-medium">Tags</th>
              <th className="pb-2 pr-4 font-medium">Pages</th>
              <th className="pb-2 pr-4 font-medium">Size</th>
              <th className="pb-2 pr-4 font-medium">Enabled</th>
              <th className="pb-2 pr-4 font-medium">Last indexed</th>
              <th className="pb-2 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sources.length === 0 && (
              <tr>
                <td colSpan={8} className="py-6 text-center text-chat-muted-foreground">
                  No sources registered. Upload a file above.
                </td>
              </tr>
            )}
            {sources.map(s => {
              const isEditing = editingFilename === s.filename;
              return (
                <tr key={s.filename} className="border-b border-chat-border/50 hover:bg-chat-muted/30">
                  <td className="py-3 pr-4 font-mono text-xs text-chat-muted-foreground max-w-[180px] truncate" title={s.filename}>
                    {s.filename}
                  </td>

                  {/* Title (editable) */}
                  <td className="py-3 pr-4">
                    {isEditing ? (
                      <input
                        className="w-full bg-chat-muted border border-chat-border rounded px-2 py-1 text-xs text-chat-foreground"
                        value={edits.title}
                        onChange={e => setEdits(p => ({ ...p, title: e.target.value }))}
                        placeholder="Title"
                      />
                    ) : (
                      <span className="text-chat-foreground">{s.title ?? <span className="text-chat-muted-foreground italic">untitled</span>}</span>
                    )}
                  </td>

                  {/* Tags (editable) */}
                  <td className="py-3 pr-4">
                    {isEditing ? (
                      <div className="flex gap-1">
                        {TAG_OPTIONS.map(tag => (
                          <button
                            key={tag}
                            onClick={() => toggleTag(tag)}
                            className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
                              edits.tags.includes(tag)
                                ? 'bg-chat-primary text-white'
                                : 'bg-chat-muted text-chat-muted-foreground hover:bg-chat-primary/20'
                            }`}
                          >
                            {tag}
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="flex gap-1 flex-wrap">
                        {s.tags.length === 0
                          ? <span className="text-chat-muted-foreground italic text-xs">none</span>
                          : s.tags.map(tag => (
                              <span key={tag} className="px-2 py-0.5 rounded text-xs font-medium bg-chat-primary/20 text-chat-primary">
                                {tag}
                              </span>
                            ))
                        }
                      </div>
                    )}
                  </td>

                  <td className="py-3 pr-4 text-chat-muted-foreground">{s.page_count ?? '—'}</td>
                  <td className="py-3 pr-4 text-chat-muted-foreground">{(s.size_bytes / 1024 / 1024).toFixed(1)} MB</td>

                  {/* Enabled toggle */}
                  <td className="py-3 pr-4">
                    {isEditing ? (
                      <input
                        type="checkbox"
                        checked={edits.enabled}
                        onChange={e => setEdits(p => ({ ...p, enabled: e.target.checked }))}
                        className="w-4 h-4 accent-chat-primary"
                      />
                    ) : (
                      <span className={s.enabled ? 'text-success' : 'text-chat-muted-foreground'}>
                        {s.enabled ? '✓' : '✗'}
                      </span>
                    )}
                  </td>

                  <td className="py-3 pr-4 text-xs text-chat-muted-foreground">
                    {s.last_indexed_at ? new Date(s.last_indexed_at).toLocaleDateString() : '—'}
                  </td>

                  {/* Action buttons */}
                  <td className="py-3">
                    {isEditing ? (
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => saveEdit(s.filename)}>Save</Button>
                        <Button size="sm" variant="ghost" onClick={() => setEditingFilename(null)}>Cancel</Button>
                      </div>
                    ) : confirmDelete === s.filename ? (
                      <div className="flex gap-2">
                        <Button size="sm" variant="danger" onClick={() => handleDelete(s.filename)}>Confirm delete</Button>
                        <Button size="sm" variant="ghost" onClick={() => setConfirmDelete(null)}>Cancel</Button>
                      </div>
                    ) : (
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline" onClick={() => startEdit(s)}>Edit</Button>
                        <Button size="sm" variant="ghost" onClick={() => setConfirmDelete(s.filename)}>
                          <span className="text-danger">Delete</span>
                        </Button>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
