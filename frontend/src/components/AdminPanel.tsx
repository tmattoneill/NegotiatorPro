/**
 * Admin Panel Component
 * Provides admin-only functionality for user/negotiation management, usage stats, and database operations
 */
import { useState, useEffect } from 'react';
import { useAuthStore } from '../store/authStore';
import * as api from '../services/api';
import './AdminPanel.css';

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

type TabView = 'users' | 'negotiations' | 'usage' | 'database';

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
      <div className="admin-panel">
        <div className="error-message">
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
    <div className="admin-panel">
      <div className="admin-header">
        <h1>🛡️ Admin Panel</h1>
        <p className="admin-subtitle">System Administration &amp; Management</p>
      </div>

      {/* Tab Navigation */}
      <div className="admin-tabs">
        <button
          className={`admin-tab ${activeTab === 'users' ? 'active' : ''}`}
          onClick={() => setActiveTab('users')}
        >
          👥 Users
        </button>
        <button
          className={`admin-tab ${activeTab === 'negotiations' ? 'active' : ''}`}
          onClick={() => setActiveTab('negotiations')}
        >
          📋 Negotiations
        </button>
        <button
          className={`admin-tab ${activeTab === 'usage' ? 'active' : ''}`}
          onClick={() => setActiveTab('usage')}
        >
          📊 Usage Stats
        </button>
        <button
          className={`admin-tab ${activeTab === 'database' ? 'active' : ''}`}
          onClick={() => setActiveTab('database')}
        >
          🗄️ Database
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="admin-error">
          ⚠️ {error}
        </div>
      )}

      {/* Loading Indicator */}
      {loading && (
        <div className="admin-loading">
          Loading...
        </div>
      )}

      {/* Tab Content */}
      <div className="admin-content">
        {/* Users Tab */}
        {activeTab === 'users' && (
          <div className="admin-users">
            <h2>User Management</h2>
            <p className="admin-note">Total users: {users.length}</p>

            <div className="users-table-container">
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>Username</th>
                    <th>Email</th>
                    <th>Role</th>
                    <th>Created</th>
                    <th>Last Login</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => (
                    <tr key={u.id} className={selectedUserId === u.id ? 'selected' : ''}>
                      <td>{u.username}</td>
                      <td>{u.email}</td>
                      <td>
                        <span className={`role-badge role-${u.role}`}>
                          {u.role}
                        </span>
                      </td>
                      <td>{formatDate(u.created_at)}</td>
                      <td>{formatDate(u.last_login)}</td>
                      <td>
                        <span className={`status-badge ${u.is_active ? 'active' : 'inactive'}`}>
                          {u.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                      <td>
                        {u.role !== 'admin' ? (
                          <button
                            className="admin-btn-danger"
                            onClick={() => handleDeleteUser(u.id, u.username)}
                            disabled={loading}
                          >
                            Delete
                          </button>
                        ) : (
                          <span className="admin-note">Protected</span>
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
          <div className="admin-negotiations">
            <h2>Negotiation Management</h2>
            <div className="admin-filters">
              <label>
                Filter by user:
                <select
                  value={negotiationFilter}
                  onChange={(e) => setNegotiationFilter(e.target.value)}
                >
                  <option value="all">All Users</option>
                  {users.map((u) => (
                    <option key={u.id} value={u.id}>
                      {u.username}
                    </option>
                  ))}
                </select>
              </label>
              <p className="admin-note">
                Showing {filteredNegotiations.length} of {negotiations.length} negotiations
              </p>
            </div>

            <div className="negotiations-table-container">
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>Title</th>
                    <th>User</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Updated</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredNegotiations.map((n) => (
                    <tr key={n.id}>
                      <td>{n.title}</td>
                      <td>{n.username}</td>
                      <td>
                        <span className={`status-badge status-${n.status}`}>
                          {n.status}
                        </span>
                      </td>
                      <td>{formatDate(n.created_at)}</td>
                      <td>{formatDate(n.updated_at)}</td>
                      <td>
                        <button
                          className="admin-btn-danger"
                          onClick={() => handleDeleteNegotiation(n.id, n.title)}
                          disabled={loading}
                        >
                          Delete
                        </button>
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
          <div className="admin-usage">
            <h2>Usage Statistics</h2>
            <p className="admin-note">Token usage and costs across all users</p>

            <div className="usage-table-container">
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Requests</th>
                    <th>Total Tokens</th>
                    <th>Total Cost</th>
                    <th>Models Used</th>
                    <th>Last Activity</th>
                  </tr>
                </thead>
                <tbody>
                  {usageStats.map((stat) => (
                    <tr key={stat.user_id}>
                      <td>{stat.username}</td>
                      <td>{stat.total_requests.toLocaleString()}</td>
                      <td>{stat.total_tokens.toLocaleString()}</td>
                      <td>{formatCurrency(stat.total_cost)}</td>
                      <td>
                        <div className="models-list">
                          {stat.models_used.length > 0
                            ? stat.models_used.join(', ')
                            : 'None'}
                        </div>
                      </td>
                      <td>{formatDate(stat.last_activity)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Summary Statistics */}
              <div className="usage-summary">
                <h3>Summary</h3>
                <div className="summary-grid">
                  <div className="summary-item">
                    <span className="summary-label">Total Requests:</span>
                    <span className="summary-value">
                      {usageStats.reduce((sum, s) => sum + s.total_requests, 0).toLocaleString()}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Total Tokens:</span>
                    <span className="summary-value">
                      {usageStats.reduce((sum, s) => sum + s.total_tokens, 0).toLocaleString()}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Total Cost:</span>
                    <span className="summary-value">
                      {formatCurrency(usageStats.reduce((sum, s) => sum + s.total_cost, 0))}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Database Tab */}
        {activeTab === 'database' && (
          <div className="admin-database">
            <h2>Database Management</h2>

            <div className="database-actions">
              <div className="danger-zone">
                <h3>⚠️ Danger Zone</h3>
                <p className="warning-text">
                  The following action will permanently delete ALL user data except admin accounts.
                  This cannot be undone!
                </p>

                {!confirmReset ? (
                  <button
                    className="admin-btn-danger-large"
                    onClick={() => setConfirmReset(true)}
                  >
                    Reset Database
                  </button>
                ) : (
                  <div className="reset-confirmation">
                    <p className="confirm-prompt">
                      Type <strong>RESET DATABASE</strong> to confirm:
                    </p>
                    <input
                      type="text"
                      className="confirm-input"
                      value={resetConfirmText}
                      onChange={(e) => setResetConfirmText(e.target.value)}
                      placeholder="RESET DATABASE"
                    />
                    <div className="confirm-buttons">
                      <button
                        className="admin-btn-danger-large"
                        onClick={handleResetDatabase}
                        disabled={loading || resetConfirmText !== 'RESET DATABASE'}
                      >
                        Confirm Reset
                      </button>
                      <button
                        className="admin-btn-secondary"
                        onClick={() => {
                          setConfirmReset(false);
                          setResetConfirmText('');
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AdminPanel;
