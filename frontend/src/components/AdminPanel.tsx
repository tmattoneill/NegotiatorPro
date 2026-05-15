/**
 * Admin Panel Component
 * Provides admin-only functionality for user/negotiation management, usage stats, and database operations
 */
import { useState, useEffect } from 'react';
import { useAuthStore } from '../store/authStore';
import * as api from '../services/api';
import Button from './ui/Button';
import Badge from './ui/Badge';
import Input from './ui/Input';
import LLMConfigTab from './LLMConfigTab';
import SourcesTab from './admin/SourcesTab';
import VectorstorePanel from './admin/VectorstorePanel';

interface User {
  id: string;
  username: string;
  email: string;
  role: string;
  created_at: string;
  last_login: string | null;
  is_active: boolean;
}

interface Negotiation {
  id: string;
  user_id: string;
  username: string;
  title: string;
  status: string;
  created_at: string;
  updated_at: string;
}

interface UsageStats {
  user_id: string;
  username: string;
  total_requests: number;
  total_tokens: number;
  total_cost: number;
  models_used: string[];
  last_activity: string | null;
}

type TabView = 'users' | 'negotiations' | 'usage' | 'llm-config' | 'database' | 'sources';

const AdminPanel = () => {
  const user = useAuthStore((state) => state.user);
  const [activeTab, setActiveTab] = useState<TabView>('users');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Users tab state
  const [users, setUsers] = useState<User[]>([]);
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);

  // Negotiations tab state
  const [negotiations, setNegotiations] = useState<Negotiation[]>([]);
  const [filteredNegotiations, setFilteredNegotiations] = useState<Negotiation[]>([]);
  const [negotiationFilter, setNegotiationFilter] = useState<string>('all');

  // Usage tab state
  const [usageStats, setUsageStats] = useState<UsageStats[]>([]);

  // Database tab state
  const [confirmReset, setConfirmReset] = useState(false);
  const [resetConfirmText, setResetConfirmText] = useState('');

  // Check if user is admin
  if (user?.role !== 'admin') {
    return (
      <div className="flex-1 overflow-y-auto p-8 bg-chat-muted">
        <div className="rounded-md border border-danger/30 bg-danger/10 text-danger px-4 py-3">
          Access Denied: Admin privileges required
        </div>
      </div>
    );
  }

  // Load data based on active tab
  useEffect(() => {
    loadTabData();
  }, [activeTab]);

  // Filter negotiations when filter changes
  useEffect(() => {
    if (negotiationFilter === 'all') {
      setFilteredNegotiations(negotiations);
    } else {
      setFilteredNegotiations(
        negotiations.filter((n) => n.user_id === negotiationFilter)
      );
    }
  }, [negotiations, negotiationFilter]);

  const loadTabData = async () => {
    setLoading(true);
    setError(null);

    try {
      switch (activeTab) {
        case 'users':
          await loadUsers();
          break;
        case 'negotiations':
          await loadNegotiations();
          break;
        case 'usage':
          await loadUsageStats();
          break;
        case 'database':
          // No data to load for database tab
          break;
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const loadUsers = async () => {
    const data = await api.adminListUsers();
    setUsers(data);
  };

  const loadNegotiations = async () => {
    const data = await api.adminListNegotiations();
    setNegotiations(data);
    setFilteredNegotiations(data);
  };

  const loadUsageStats = async () => {
    const data = await api.adminGetUsageSummary();
    setUsageStats(data);
  };

  const handleDeleteUser = async (userId: string, username: string) => {
    if (!window.confirm(`Are you sure you want to delete user "${username}"? This will also delete all their negotiations and conversations.`)) {
      return;
    }

    try {
      setLoading(true);
      await api.adminDeleteUser(userId);
      await loadUsers();
      setSelectedUserId(null);
      alert(`User "${username}" deleted successfully`);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to delete user');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteNegotiation = async (negotiationId: string, title: string) => {
    if (!window.confirm(`Are you sure you want to delete negotiation "${title}"? This will also delete all associated conversations.`)) {
      return;
    }

    try {
      setLoading(true);
      await api.adminDeleteNegotiation(negotiationId);
      await loadNegotiations();
      alert(`Negotiation "${title}" deleted successfully`);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to delete negotiation');
    } finally {
      setLoading(false);
    }
  };

  const handleResetDatabase = async () => {
    if (resetConfirmText !== 'RESET DATABASE') {
      alert('Please type "RESET DATABASE" to confirm');
      return;
    }

    try {
      setLoading(true);
      const result = await api.adminResetDatabase();
      alert(
        `Database reset successfully!\n\n` +
        `Users deleted: ${result.users_deleted}\n` +
        `Negotiations deleted: ${result.negotiations_deleted}\n` +
        `Conversations deleted: ${result.conversations_deleted}\n` +
        `Messages deleted: ${result.messages_deleted}`
      );
      setConfirmReset(false);
      setResetConfirmText('');
      // Reload all data
      await loadTabData();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to reset database');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return 'Never';
    return new Date(dateStr).toLocaleString();
  };

  const formatCurrency = (amount: number) => {
    return `$${amount.toFixed(4)}`;
  };

  return (
    <div className="flex-1 overflow-y-auto p-8 bg-chat-muted">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-chat-foreground">🛡️ Admin Panel</h1>
        <p className="text-sm text-chat-muted-foreground">System Administration &amp; Management</p>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 mb-6 border-b-2 border-chat-border">
        <button
          className={`px-4 py-2 -mb-[2px] border-b-4 ${activeTab === 'users' ? 'border-chat-primary text-chat-primary font-semibold bg-chat-primary/10' : 'border-transparent text-chat-muted-foreground hover:bg-chat-muted hover:text-chat-foreground'}`}
          onClick={() => setActiveTab('users')}
        >
          👥 Users
        </button>
        <button
          className={`px-4 py-2 -mb-[2px] border-b-4 ${activeTab === 'negotiations' ? 'border-chat-primary text-chat-primary font-semibold bg-chat-primary/10' : 'border-transparent text-chat-muted-foreground hover:bg-chat-muted hover:text-chat-foreground'}`}
          onClick={() => setActiveTab('negotiations')}
        >
          📋 Negotiations
        </button>
        <button
          className={`px-4 py-2 -mb-[2px] border-b-4 ${activeTab === 'usage' ? 'border-chat-primary text-chat-primary font-semibold bg-chat-primary/10' : 'border-transparent text-chat-muted-foreground hover:bg-chat-muted hover:text-chat-foreground'}`}
          onClick={() => setActiveTab('usage')}
        >
          📊 Usage Stats
        </button>
        <button
          className={`px-4 py-2 -mb-[2px] border-b-4 ${activeTab === 'llm-config' ? 'border-chat-primary text-chat-primary font-semibold bg-chat-primary/10' : 'border-transparent text-chat-muted-foreground hover:bg-chat-muted hover:text-chat-foreground'}`}
          onClick={() => setActiveTab('llm-config')}
        >
          🤖 LLM Config
        </button>
        <button
          className={`px-4 py-2 -mb-[2px] border-b-4 ${activeTab === 'database' ? 'border-chat-primary text-chat-primary font-semibold bg-chat-primary/10' : 'border-transparent text-chat-muted-foreground hover:bg-chat-muted hover:text-chat-foreground'}`}
          onClick={() => setActiveTab('database')}
        >
          🗄️ Database
        </button>
        <button
          className={`px-4 py-2 -mb-[2px] border-b-4 ${activeTab === 'sources' ? 'border-chat-primary text-chat-primary font-semibold bg-chat-primary/10' : 'border-transparent text-chat-muted-foreground hover:bg-chat-muted hover:text-chat-foreground'}`}
          onClick={() => setActiveTab('sources')}
        >
          📚 Sources &amp; RAG
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger mb-4">⚠️ {error}</div>
      )}

      {/* Loading Indicator */}
      {loading && (
        <div className="p-8 text-center text-chat-muted-foreground">Loading...</div>
      )}

      {/* Tab Content */}
      <div className="bg-chat-card rounded-lg p-6 md:p-8 shadow-card">
        {/* Users Tab */}
        {activeTab === 'users' && (
          <div>
            <h2 className="text-xl font-semibold text-chat-foreground">User Management</h2>
            <p className="text-sm text-chat-muted-foreground mb-4">Total users: {users.length}</p>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-separate border-spacing-0">
                <thead className="bg-chat-sidebar text-white">
                  <tr>
                    <th className="p-3">Username</th>
                    <th className="p-3">Email</th>
                    <th className="p-3">Role</th>
                    <th className="p-3">Created</th>
                    <th className="p-3">Last Login</th>
                    <th className="p-3">Status</th>
                    <th className="p-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => (
                    <tr key={u.id} className={`${selectedUserId === u.id ? 'bg-chat-primary/10' : ''} hover:bg-chat-muted`}> 
                      <td className="p-3">{u.username}</td>
                      <td className="p-3">{u.email}</td>
                      <td className="p-3">
                        <Badge variant="outline" className="uppercase text-[11px]">{u.role}</Badge>
                      </td>
                      <td className="p-3">{formatDate(u.created_at)}</td>
                      <td className="p-3">{formatDate(u.last_login)}</td>
                      <td className="p-3">
                        {u.is_active ? (
                          <Badge variant="success">Active</Badge>
                        ) : (
                          <Badge variant="muted">Inactive</Badge>
                        )}
                      </td>
                      <td className="p-3">
                        {u.role !== 'admin' ? (
                          <Button variant="danger" size="sm" onClick={() => handleDeleteUser(u.id, u.username)} disabled={loading}>
                            Delete
                          </Button>
                        ) : (
                          <span className="text-sm text-chat-muted-foreground">Protected</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Negotiations Tab */}
        {activeTab === 'negotiations' && (
          <div>
            <h2 className="text-xl font-semibold text-chat-foreground">Negotiation Management</h2>
            <div className="flex items-center gap-4 my-4 p-4 bg-chat-muted rounded">
              <label className="font-medium flex items-center gap-2">
                Filter by user:
                <select
                  value={negotiationFilter}
                  onChange={(e) => setNegotiationFilter(e.target.value)}
                  className="px-3 py-2 text-sm border border-chat-border rounded outline-none focus:ring-2 focus:ring-chat-primary/20 focus:border-chat-primary"
                >
                  <option value="all">All Users</option>
                  {users.map((u) => (
                    <option key={u.id} value={u.id}>
                      {u.username}
                    </option>
                  ))}
                </select>
              </label>
              <p className="text-sm text-chat-muted-foreground">
                Showing {filteredNegotiations.length} of {negotiations.length} negotiations
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-separate border-spacing-0">
                <thead className="bg-chat-sidebar text-white">
                  <tr>
                    <th className="p-3">Title</th>
                    <th className="p-3">User</th>
                    <th className="p-3">Status</th>
                    <th className="p-3">Created</th>
                    <th className="p-3">Updated</th>
                    <th className="p-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredNegotiations.map((n) => (
                    <tr key={n.id} className="hover:bg-chat-muted">
                      <td className="p-3">{n.title}</td>
                      <td className="p-3">{n.username}</td>
                      <td className="p-3">
                        <Badge variant={n.status === 'active' || n.status === 'won' ? 'success' : 'muted'}>{n.status}</Badge>
                      </td>
                      <td className="p-3">{formatDate(n.created_at)}</td>
                      <td className="p-3">{formatDate(n.updated_at)}</td>
                      <td className="p-3">
                        <Button variant="danger" size="sm" onClick={() => handleDeleteNegotiation(n.id, n.title)} disabled={loading}>Delete</Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Usage Stats Tab */}
        {activeTab === 'usage' && (
          <div>
            <h2 className="text-xl font-semibold text-chat-foreground">Usage Statistics</h2>
            <p className="text-sm text-chat-muted-foreground">Token usage and costs across all users</p>

            <div className="overflow-x-auto mt-4">
              <table className="w-full text-left border-separate border-spacing-0">
                <thead>
                  <tr>
                    <th className="p-3">User</th>
                    <th className="p-3">Requests</th>
                    <th className="p-3">Total Tokens</th>
                    <th className="p-3">Total Cost</th>
                    <th className="p-3">Models Used</th>
                    <th className="p-3">Last Activity</th>
                  </tr>
                </thead>
                <tbody>
                  {usageStats.map((stat) => (
                    <tr key={stat.user_id} className="hover:bg-chat-muted">
                      <td className="p-3">{stat.username}</td>
                      <td className="p-3">{stat.total_requests.toLocaleString()}</td>
                      <td className="p-3">{stat.total_tokens.toLocaleString()}</td>
                      <td className="p-3">{formatCurrency(stat.total_cost)}</td>
                      <td className="p-3">
                        <div className="text-sm text-chat-muted-foreground">
                          {stat.models_used.length > 0
                            ? stat.models_used.join(', ')
                            : 'None'}
                        </div>
                      </td>
                      <td className="p-3">{formatDate(stat.last_activity)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Summary Statistics */}
              <div className="mt-8 p-6 bg-chat-muted rounded-lg">
                <h3 className="text-lg font-semibold mb-4">Summary</h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="rounded border border-chat-border p-4 bg-white">
                    <span className="block text-sm text-chat-muted-foreground">Total Requests:</span>
                    <span className="text-xl font-semibold">
                      {usageStats.reduce((sum, s) => sum + s.total_requests, 0).toLocaleString()}
                    </span>
                  </div>
                  <div className="rounded border border-chat-border p-4 bg-white">
                    <span className="block text-sm text-chat-muted-foreground">Total Tokens:</span>
                    <span className="text-xl font-semibold">
                      {usageStats.reduce((sum, s) => sum + s.total_tokens, 0).toLocaleString()}
                    </span>
                  </div>
                  <div className="rounded border border-chat-border p-4 bg-white">
                    <span className="block text-sm text-chat-muted-foreground">Total Cost:</span>
                    <span className="text-xl font-semibold">
                      {formatCurrency(usageStats.reduce((sum, s) => sum + s.total_cost, 0))}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* LLM Config Tab */}
        {activeTab === 'llm-config' && <LLMConfigTab />}

        {/* Database Tab */}
        {activeTab === 'database' && (
          <div>
            <h2 className="text-xl font-semibold text-chat-foreground">Database Management</h2>

            <div className="mt-4">
              <div className="rounded-lg border border-danger/30 bg-danger/5 p-6">
                <h3 className="text-lg font-semibold mb-2">⚠️ Danger Zone</h3>
                <p className="text-sm text-chat-muted-foreground mb-4">
                  The following action will permanently delete ALL user data except admin accounts.
                  This cannot be undone!
                </p>

                {!confirmReset ? (
                  <Button variant="danger" size="lg" onClick={() => setConfirmReset(true)}>
                    Reset Database
                  </Button>
                ) : (
                  <div className="mt-3">
                    <p className="text-sm mb-2">
                      Type <strong>RESET DATABASE</strong> to confirm:
                    </p>
                    <Input
                      type="text"
                      value={resetConfirmText}
                      onChange={(e) => setResetConfirmText(e.target.value)}
                      placeholder="RESET DATABASE"
                      className="max-w-sm"
                    />
                    <div className="flex gap-3 mt-3">
                      <Button
                        variant="danger"
                        onClick={handleResetDatabase}
                        disabled={loading || resetConfirmText !== 'RESET DATABASE'}
                      >
                        Confirm Reset
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setConfirmReset(false);
                          setResetConfirmText('');
                        }}
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Sources & RAG Tab */}
        {activeTab === 'sources' && (
          <div className="space-y-8">
            <div>
              <h2 className="text-xl font-semibold text-chat-foreground mb-1">Source Documents</h2>
              <p className="text-sm text-chat-muted-foreground">Upload, tag, and manage the knowledge base used by the RAG system.</p>
            </div>
            <SourcesTab />
            <hr className="border-chat-border" />
            <div>
              <h2 className="text-xl font-semibold text-chat-foreground mb-1">Vectorstore</h2>
              <p className="text-sm text-chat-muted-foreground">Rebuild the FAISS index from enabled sources and test retrieval quality.</p>
            </div>
            <VectorstorePanel />
          </div>
        )}
      </div>
    </div>
  );
};

export default AdminPanel;
