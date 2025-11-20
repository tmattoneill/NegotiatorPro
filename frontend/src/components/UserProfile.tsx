/**
 * UserProfile Component
 *
 * View and edit user profile information including API keys
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import api from '../services/api';

export default function UserProfile() {
  const navigate = useNavigate();
  const { user, updateUser } = useAuthStore();
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [testingOpenAI, setTestingOpenAI] = useState(false);
  const [testingAnthropic, setTestingAnthropic] = useState(false);
  const [openAITestResult, setOpenAITestResult] = useState<{ valid: boolean; message: string } | null>(null);
  const [anthropicTestResult, setAnthropicTestResult] = useState<{ valid: boolean; message: string } | null>(null);

  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
    email: '',
    openai_api_key: '',
    anthropic_api_key: '',
  });

  useEffect(() => {
    if (user) {
      setFormData({
        first_name: user.first_name || '',
        last_name: user.last_name || '',
        email: user.email || '',
        openai_api_key: '',
        anthropic_api_key: '',
      });
    }
  }, [user]);

  const handleTestAPIKey = async (provider: 'openai' | 'anthropic') => {
    const apiKey = provider === 'openai' ? formData.openai_api_key : formData.anthropic_api_key;

    if (!apiKey || !apiKey.trim()) {
      if (provider === 'openai') {
        setOpenAITestResult({ valid: false, message: 'Please enter an API key to test' });
      } else {
        setAnthropicTestResult({ valid: false, message: 'Please enter an API key to test' });
      }
      return;
    }

    if (provider === 'openai') {
      setTestingOpenAI(true);
      setOpenAITestResult(null);
    } else {
      setTestingAnthropic(true);
      setAnthropicTestResult(null);
    }

    try {
      const response = await api.post('/users/test-api-key', {
        provider,
        api_key: apiKey,
      });

      if (provider === 'openai') {
        setOpenAITestResult({ valid: true, message: response.data.message });
      } else {
        setAnthropicTestResult({ valid: true, message: response.data.message });
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'API key validation failed';
      if (provider === 'openai') {
        setOpenAITestResult({ valid: false, message: errorMessage });
      } else {
        setAnthropicTestResult({ valid: false, message: errorMessage });
      }
    } finally {
      if (provider === 'openai') {
        setTestingOpenAI(false);
      } else {
        setTestingAnthropic(false);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setIsSaving(true);

    try {
      const updateData: any = {
        first_name: formData.first_name || undefined,
        last_name: formData.last_name || undefined,
        email: formData.email,
      };

      // Only include API keys if they were changed (non-empty)
      if (formData.openai_api_key) {
        updateData.openai_api_key = formData.openai_api_key;
      }
      if (formData.anthropic_api_key) {
        updateData.anthropic_api_key = formData.anthropic_api_key;
      }

      const response = await api.put(`/users/${user?.id}`, updateData);

      updateUser(response.data);
      setSuccess('Profile updated successfully!');
      setIsEditing(false);

      // Clear API key fields after successful update
      setFormData(prev => ({
        ...prev,
        openai_api_key: '',
        anthropic_api_key: '',
      }));

      setTimeout(() => setSuccess(''), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update profile');
    } finally {
      setIsSaving(false);
    }
  };

  if (!user) {
    return null;
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#efefef',
      padding: '40px 20px',
    }}>
      <div style={{
        maxWidth: '800px',
        margin: '0 auto',
      }}>
        {/* Header */}
        <div style={{
          marginBottom: '32px',
        }}>
          <button
            onClick={() => navigate('/')}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#3498db',
              fontSize: '14px',
              cursor: 'pointer',
              marginBottom: '16px',
              padding: '0',
            }}
          >
            ← Back to Chat
          </button>
          <h1 style={{
            fontSize: '32px',
            fontWeight: '600',
            color: '#191919',
            margin: '0 0 8px 0',
          }}>
            User Profile
          </h1>
          <p style={{
            fontSize: '16px',
            color: '#9fadbd',
            margin: '0',
          }}>
            Manage your account settings and API keys
          </p>
        </div>

        {/* Profile Card */}
        <div style={{
          background: '#ffffff',
          borderRadius: '8px',
          padding: '32px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        }}>
          {/* Account Information */}
          <div style={{ marginBottom: '32px' }}>
            <h2 style={{
              fontSize: '18px',
              fontWeight: '600',
              color: '#191919',
              marginBottom: '16px',
            }}>
              Account Information
            </h2>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '150px 1fr',
              gap: '12px',
              fontSize: '14px',
              marginBottom: '20px',
            }}>
              <div style={{ color: '#9fadbd' }}>Username:</div>
              <div style={{ color: '#191919', fontWeight: '500' }}>{user.username}</div>

              <div style={{ color: '#9fadbd' }}>Role:</div>
              <div style={{ color: '#191919', fontWeight: '500' }}>{user.role}</div>
            </div>

            {/* API Keys Status */}
            <div style={{ marginTop: '20px' }}>
              <h3 style={{
                fontSize: '14px',
                fontWeight: '600',
                color: '#191919',
                marginBottom: '12px',
              }}>
                API Keys Status
              </h3>
              <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 14px',
                  borderRadius: '6px',
                  background: user.has_openai_key ? '#f0fdf4' : '#f8f9fa',
                  border: user.has_openai_key ? '1px solid #86efac' : '1px solid #e0e0e0',
                }}>
                  <div style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: user.has_openai_key ? '#22c55e' : '#9fadbd',
                  }}></div>
                  <span style={{
                    fontSize: '13px',
                    fontWeight: '500',
                    color: user.has_openai_key ? '#166534' : '#9fadbd',
                  }}>
                    OpenAI
                  </span>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 14px',
                  borderRadius: '6px',
                  background: user.has_anthropic_key ? '#f0fdf4' : '#f8f9fa',
                  border: user.has_anthropic_key ? '1px solid #86efac' : '1px solid #e0e0e0',
                }}>
                  <div style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: user.has_anthropic_key ? '#22c55e' : '#9fadbd',
                  }}></div>
                  <span style={{
                    fontSize: '13px',
                    fontWeight: '500',
                    color: user.has_anthropic_key ? '#166534' : '#9fadbd',
                  }}>
                    Anthropic
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Edit Form */}
          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '32px' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px',
              }}>
                <h2 style={{
                  fontSize: '18px',
                  fontWeight: '600',
                  color: '#191919',
                  margin: '0',
                }}>
                  Profile Details
                </h2>
                {!isEditing && (
                  <button
                    type="button"
                    onClick={() => setIsEditing(true)}
                    style={{
                      background: '#3498db',
                      color: '#ffffff',
                      border: 'none',
                      borderRadius: '6px',
                      padding: '8px 16px',
                      fontSize: '14px',
                      cursor: 'pointer',
                      fontWeight: '500',
                    }}
                  >
                    Edit Profile
                  </button>
                )}
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div>
                  <label style={{
                    display: 'block',
                    fontSize: '14px',
                    fontWeight: '500',
                    color: '#191919',
                    marginBottom: '6px',
                  }}>
                    First Name
                  </label>
                  <input
                    type="text"
                    value={formData.first_name}
                    onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
                    disabled={!isEditing}
                    style={{
                      width: '100%',
                      padding: '10px 12px',
                      fontSize: '14px',
                      border: '1px solid #9fadbd',
                      borderRadius: '6px',
                      background: isEditing ? '#ffffff' : '#efefef',
                      color: '#191919',
                    }}
                  />
                </div>

                <div>
                  <label style={{
                    display: 'block',
                    fontSize: '14px',
                    fontWeight: '500',
                    color: '#191919',
                    marginBottom: '6px',
                  }}>
                    Last Name
                  </label>
                  <input
                    type="text"
                    value={formData.last_name}
                    onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
                    disabled={!isEditing}
                    style={{
                      width: '100%',
                      padding: '10px 12px',
                      fontSize: '14px',
                      border: '1px solid #9fadbd',
                      borderRadius: '6px',
                      background: isEditing ? '#ffffff' : '#efefef',
                      color: '#191919',
                    }}
                  />
                </div>

                <div>
                  <label style={{
                    display: 'block',
                    fontSize: '14px',
                    fontWeight: '500',
                    color: '#191919',
                    marginBottom: '6px',
                  }}>
                    Email
                  </label>
                  <input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    disabled={!isEditing}
                    required
                    style={{
                      width: '100%',
                      padding: '10px 12px',
                      fontSize: '14px',
                      border: '1px solid #9fadbd',
                      borderRadius: '6px',
                      background: isEditing ? '#ffffff' : '#efefef',
                      color: '#191919',
                    }}
                  />
                </div>
              </div>
            </div>

            {/* API Keys Section */}
            {isEditing && (
              <div style={{ marginBottom: '24px' }}>
                <h2 style={{
                  fontSize: '18px',
                  fontWeight: '600',
                  color: '#191919',
                  marginBottom: '8px',
                }}>
                  API Keys
                </h2>
                <p style={{
                  fontSize: '13px',
                  color: '#9fadbd',
                  marginBottom: '16px',
                }}>
                  Leave blank to keep existing keys unchanged
                </p>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {/* OpenAI API Key */}
                  <div>
                    <label style={{
                      display: 'block',
                      fontSize: '14px',
                      fontWeight: '500',
                      color: '#191919',
                      marginBottom: '6px',
                    }}>
                      OpenAI API Key
                    </label>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <input
                        type="password"
                        value={formData.openai_api_key}
                        onChange={(e) => {
                          setFormData({ ...formData, openai_api_key: e.target.value });
                          setOpenAITestResult(null);
                        }}
                        placeholder="sk-..."
                        style={{
                          flex: 1,
                          padding: '10px 12px',
                          fontSize: '14px',
                          border: '1px solid #9fadbd',
                          borderRadius: '6px',
                          background: '#ffffff',
                          color: '#191919',
                        }}
                      />
                      <button
                        type="button"
                        onClick={() => handleTestAPIKey('openai')}
                        disabled={testingOpenAI || !formData.openai_api_key}
                        style={{
                          padding: '10px 16px',
                          fontSize: '14px',
                          fontWeight: '500',
                          color: '#ffffff',
                          background: testingOpenAI ? '#9fadbd' : '#3498db',
                          border: 'none',
                          borderRadius: '6px',
                          cursor: testingOpenAI || !formData.openai_api_key ? 'not-allowed' : 'pointer',
                          opacity: testingOpenAI || !formData.openai_api_key ? 0.6 : 1,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {testingOpenAI ? 'Testing...' : 'Test Key'}
                      </button>
                    </div>
                    {openAITestResult && (
                      <div style={{
                        marginTop: '8px',
                        padding: '8px 12px',
                        borderRadius: '6px',
                        fontSize: '13px',
                        background: openAITestResult.valid ? '#f0fdf4' : '#fff5f5',
                        border: openAITestResult.valid ? '1px solid #86efac' : '1px solid #fc8181',
                        color: openAITestResult.valid ? '#166534' : '#c53030',
                      }}>
                        {openAITestResult.message}
                      </div>
                    )}
                  </div>

                  {/* Anthropic API Key */}
                  <div>
                    <label style={{
                      display: 'block',
                      fontSize: '14px',
                      fontWeight: '500',
                      color: '#191919',
                      marginBottom: '6px',
                    }}>
                      Anthropic API Key
                    </label>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <input
                        type="password"
                        value={formData.anthropic_api_key}
                        onChange={(e) => {
                          setFormData({ ...formData, anthropic_api_key: e.target.value });
                          setAnthropicTestResult(null);
                        }}
                        placeholder="sk-ant-..."
                        style={{
                          flex: 1,
                          padding: '10px 12px',
                          fontSize: '14px',
                          border: '1px solid #9fadbd',
                          borderRadius: '6px',
                          background: '#ffffff',
                          color: '#191919',
                        }}
                      />
                      <button
                        type="button"
                        onClick={() => handleTestAPIKey('anthropic')}
                        disabled={testingAnthropic || !formData.anthropic_api_key}
                        style={{
                          padding: '10px 16px',
                          fontSize: '14px',
                          fontWeight: '500',
                          color: '#ffffff',
                          background: testingAnthropic ? '#9fadbd' : '#3498db',
                          border: 'none',
                          borderRadius: '6px',
                          cursor: testingAnthropic || !formData.anthropic_api_key ? 'not-allowed' : 'pointer',
                          opacity: testingAnthropic || !formData.anthropic_api_key ? 0.6 : 1,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {testingAnthropic ? 'Testing...' : 'Test Key'}
                      </button>
                    </div>
                    {anthropicTestResult && (
                      <div style={{
                        marginTop: '8px',
                        padding: '8px 12px',
                        borderRadius: '6px',
                        fontSize: '13px',
                        background: anthropicTestResult.valid ? '#f0fdf4' : '#fff5f5',
                        border: anthropicTestResult.valid ? '1px solid #86efac' : '1px solid #fc8181',
                        color: anthropicTestResult.valid ? '#166534' : '#c53030',
                      }}>
                        {anthropicTestResult.message}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Messages */}
            {error && (
              <div style={{
                padding: '12px 16px',
                background: '#fff5f5',
                border: '1px solid #fc8181',
                borderRadius: '6px',
                color: '#c53030',
                fontSize: '14px',
                marginBottom: '16px',
              }}>
                {error}
              </div>
            )}

            {success && (
              <div style={{
                padding: '12px 16px',
                background: '#f0fdf4',
                border: '1px solid #86efac',
                borderRadius: '6px',
                color: '#166534',
                fontSize: '14px',
                marginBottom: '16px',
              }}>
                {success}
              </div>
            )}

            {/* Action Buttons */}
            {isEditing && (
              <div style={{
                display: 'flex',
                gap: '12px',
                justifyContent: 'flex-end',
              }}>
                <button
                  type="button"
                  onClick={() => {
                    setIsEditing(false);
                    setError('');
                    // Reset form
                    setFormData({
                      first_name: user.first_name || '',
                      last_name: user.last_name || '',
                      email: user.email || '',
                      openai_api_key: '',
                      anthropic_api_key: '',
                    });
                  }}
                  style={{
                    background: '#efefef',
                    color: '#191919',
                    border: 'none',
                    borderRadius: '6px',
                    padding: '10px 20px',
                    fontSize: '14px',
                    cursor: 'pointer',
                    fontWeight: '500',
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSaving}
                  style={{
                    background: '#b68947',
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: '6px',
                    padding: '10px 20px',
                    fontSize: '14px',
                    cursor: isSaving ? 'not-allowed' : 'pointer',
                    fontWeight: '500',
                    opacity: isSaving ? 0.6 : 1,
                  }}
                >
                  {isSaving ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            )}
          </form>
        </div>
      </div>
    </div>
  );
}
