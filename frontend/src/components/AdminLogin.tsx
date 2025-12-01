/**
 * Admin Login component - Password-only admin login
 */
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { login as apiLogin } from '../services/api';

function AdminLogin() {
  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);

  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!password) {
      setError('Password is required');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Login via admin password endpoint
      const response = await apiLogin({ password });

      // Store JWT token in localStorage
      localStorage.setItem('token', response.access_token);

      // Store admin user data in auth store
      // The admin login endpoint returns a token with role="admin"
      login({
        id: 'admin',
        username: 'admin',
        email: 'admin@system',
        role: 'admin',
        is_super_admin: false,
      });

      // Redirect to main app
      navigate('/app');
    } catch (error: any) {
      console.error('Admin login failed:', error);
      const errorMessage = error.response?.data?.detail || 'Invalid admin password';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="create-profile-container">
      <div className="create-profile-card" style={{ maxWidth: '400px' }}>
        <div style={{ textAlign: 'center', marginBottom: '24px' }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            background: 'rgba(182, 137, 71, 0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 16px',
            fontSize: '28px'
          }}>
            🛡️
          </div>
          <h1 style={{ marginBottom: '8px' }}>Admin Access</h1>
          <p className="subtitle">Enter the admin password to continue</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-section">
            <div className="form-group">
              <label htmlFor="password">Admin Password</label>
              <input
                type="password"
                id="password"
                name="password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError(null);
                }}
                required
                placeholder="Enter admin password"
                autoFocus
                style={{ fontSize: '16px' }}
              />
            </div>
          </div>

          {error && (
            <div className="error-banner" style={{ marginBottom: '16px' }}>
              {error}
            </div>
          )}

          <button
            type="submit"
            className="submit-button"
            disabled={isLoading}
          >
            {isLoading ? 'Authenticating...' : 'Access Admin Panel'}
          </button>

          <div style={{ marginTop: '24px', textAlign: 'center' }}>
            <Link to="/login" style={{ color: '#9fadbd', textDecoration: 'none', fontSize: '14px' }}>
              ← Back to user login
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}

export default AdminLogin;
