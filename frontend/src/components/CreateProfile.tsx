/**
 * CreateProfile component - User registration form
 */
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import api from '../services/api';

interface FormData {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
  first_name: string;
  last_name: string;
  openai_api_key?: string;
  anthropic_api_key?: string;
}

interface FormErrors {
  username?: string;
  email?: string;
  password?: string;
  confirmPassword?: string;
  submit?: string;
}

function CreateProfile() {
  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);

  const [formData, setFormData] = useState<FormData>({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    first_name: '',
    last_name: '',
    openai_api_key: '',
    anthropic_api_key: '',
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [isLoading, setIsLoading] = useState(false);
  const [showApiKeys, setShowApiKeys] = useState(false);

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    if (formData.username.length < 3) {
      newErrors.username = 'Username must be at least 3 characters';
    }

    if (!formData.email.includes('@')) {
      newErrors.email = 'Invalid email address';
    }

    if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }

    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
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
      // Create user profile
      const response = await api.post('/users/', {
        username: formData.username,
        email: formData.email,
        password: formData.password,
        first_name: formData.first_name || undefined,
        last_name: formData.last_name || undefined,
        openai_api_key: formData.openai_api_key || undefined,
        anthropic_api_key: formData.anthropic_api_key || undefined,
        role: 'user',
      });

      // Clear any previous onboarding flags to ensure new user gets the wizard
      localStorage.removeItem('onboardingCompleted');
      localStorage.removeItem('onboardingDismissed');

      // Log the user in
      login(response.data);

      // Redirect to main app - onboarding wizard will trigger automatically
      navigate('/app');
    } catch (error: any) {
      console.error('Profile creation failed:', error);
      setErrors({
        submit: error.response?.data?.detail || 'Failed to create profile. Please try again.',
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
        <h1>Create Your Profile</h1>
        <p className="subtitle">Welcome to NegotiatorPro! Let's get you set up.</p>

        <form onSubmit={handleSubmit}>
          {/* Basic Information */}
          <div className="form-section">
            <h2>Basic Information</h2>

            <div className="form-group">
              <label htmlFor="username">Username *</label>
              <input
                type="text"
                id="username"
                name="username"
                value={formData.username}
                onChange={handleChange}
                required
                minLength={3}
                placeholder="Choose a username"
              />
              {errors.username && <span className="error">{errors.username}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="email">Email *</label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="your.email@example.com"
              />
              {errors.email && <span className="error">{errors.email}</span>}
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="first_name">First Name</label>
                <input
                  type="text"
                  id="first_name"
                  name="first_name"
                  value={formData.first_name}
                  onChange={handleChange}
                  placeholder="John"
                />
              </div>

              <div className="form-group">
                <label htmlFor="last_name">Last Name</label>
                <input
                  type="text"
                  id="last_name"
                  name="last_name"
                  value={formData.last_name}
                  onChange={handleChange}
                  placeholder="Doe"
                />
              </div>
            </div>
          </div>

          {/* Password */}
          <div className="form-section">
            <h2>Security</h2>

            <div className="form-group">
              <label htmlFor="password">Password *</label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                minLength={8}
                placeholder="At least 8 characters"
              />
              {errors.password && <span className="error">{errors.password}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm Password *</label>
              <input
                type="password"
                id="confirmPassword"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
                placeholder="Re-enter your password"
              />
              {errors.confirmPassword && <span className="error">{errors.confirmPassword}</span>}
            </div>
          </div>

          {/* API Keys (Optional) */}
          <div className="form-section">
            <h2>
              API Keys{' '}
              <span className="optional">(Optional - can be added later)</span>
            </h2>

            <button
              type="button"
              className="toggle-button"
              onClick={() => setShowApiKeys(!showApiKeys)}
            >
              {showApiKeys ? '▼' : '▶'} {showApiKeys ? 'Hide' : 'Show'} API Key Fields
            </button>

            {showApiKeys && (
              <>
                <div className="form-group">
                  <label htmlFor="openai_api_key">OpenAI API Key</label>
                  <input
                    type="password"
                    id="openai_api_key"
                    name="openai_api_key"
                    value={formData.openai_api_key}
                    onChange={handleChange}
                    placeholder="sk-..."
                  />
                  <small>For GPT-4, GPT-4o, and other OpenAI models</small>
                </div>

                <div className="form-group">
                  <label htmlFor="anthropic_api_key">Anthropic API Key</label>
                  <input
                    type="password"
                    id="anthropic_api_key"
                    name="anthropic_api_key"
                    value={formData.anthropic_api_key}
                    onChange={handleChange}
                    placeholder="sk-ant-..."
                  />
                  <small>For Claude models</small>
                </div>
              </>
            )}
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
            {isLoading ? 'Creating Profile...' : 'Create Profile'}
          </button>

          <div style={{ marginTop: '24px', textAlign: 'center', color: '#9fadbd' }}>
            Already have an account?{' '}
            <Link to="/login" style={{ color: '#3498db', textDecoration: 'none', fontWeight: 500 }}>
              Login here
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}

export default CreateProfile;
