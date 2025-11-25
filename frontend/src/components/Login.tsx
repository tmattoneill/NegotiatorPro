/**
 * Login component - User login form
 */
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import api from '../services/api';

interface FormData {
  username: string;
  password: string;
}

interface FormErrors {
  username?: string;
  password?: string;
  submit?: string;
}

function Login() {
  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);

  const [formData, setFormData] = useState<FormData>({
    username: '',
    password: '',
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [isLoading, setIsLoading] = useState(false);

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    if (formData.username.length < 3) {
      newErrors.username = 'Username must be at least 3 characters';
    }

    if (formData.password.length < 1) {
      newErrors.password = 'Password is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    setErrors({});

    try {
      // Login user
      const response = await api.post('/auth/user-login', {
        username: formData.username,
        password: formData.password,
      });

      // Store JWT token in localStorage
      localStorage.setItem('token', response.data.access_token);

      // Store user data in auth store (include is_super_admin from response)
      const userData = {
        ...response.data.user,
        is_super_admin: response.data.is_super_admin || response.data.user.is_super_admin || false,
      };
      login(userData);

      // Redirect to main app
      navigate('/app');
    } catch (error: any) {
      console.error('Login failed:', error);
      const errorMessage = error.response?.data?.detail || 'Login failed. Please check your credentials.';
      setErrors({
        submit: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    // Clear error for this field
    if (errors[name as keyof FormErrors]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
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
              {errors.username && <span className="error">{errors.username}</span>}
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
              {errors.password && <span className="error">{errors.password}</span>}
            </div>
          </div>

          {/* Submit */}
          {errors.submit && (
            <div className="error-banner">
              {errors.submit}
            </div>
          )}

          <button
            type="submit"
            className="submit-button"
            disabled={isLoading}
          >
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
