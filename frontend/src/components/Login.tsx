import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { login as apiLogin } from '../services/api';

interface FormData {
  username: string;
  password: string;
}

function Login() {
  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);

  const [formData, setFormData] = useState<FormData>({ username: '', password: '' });
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (formData.username.length < 3) {
      setError('Username must be at least 3 characters');
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiLogin({ username: formData.username, password: formData.password });
      localStorage.setItem('token', response.access_token);
      login(response.user);
      navigate('/app');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setError(null);
  };

  return (
    <div className="create-profile-container">
      <div className="create-profile-card">
        <h1>Welcome Back</h1>
        <p className="subtitle">Login to continue to NegotiatorPro</p>

        <form onSubmit={handleSubmit}>
          <div className="form-section">
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input
                type="text"
                id="username"
                name="username"
                value={formData.username}
                onChange={handleChange}
                required
                minLength={3}
                placeholder="Enter your username"
                autoFocus
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                placeholder="Enter your password"
              />
            </div>
          </div>

          {error && <div className="error-banner">{error}</div>}

          <button type="submit" className="submit-button" disabled={isLoading}>
            {isLoading ? 'Logging in...' : 'Login'}
          </button>

          <div style={{ marginTop: '24px', textAlign: 'center', color: '#9fadbd' }}>
            Don't have an account?{' '}
            <Link to="/create-profile" style={{ color: '#3498db', textDecoration: 'none', fontWeight: 500 }}>
              Create one here
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Login;
