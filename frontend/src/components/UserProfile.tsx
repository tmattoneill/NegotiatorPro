/**
 * UserProfile Component
 *
 * View and edit user profile information including API keys
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { usePersonaStore } from '../store/personaStore';
import api from '../services/api';
import type { UserPersona, UserPersonaCreate } from '../types/personas';
import Button from './ui/Button';
import Input from './ui/Input';
import Textarea from './ui/Textarea';
import Badge from './ui/Badge';
import ProviderSelector from './ProviderSelector';

export default function UserProfile() {
  const navigate = useNavigate();
  const { user, updateUser } = useAuthStore();
  const {
    userPersonas,
    fetchUserPersonas,
    createUserPersona,
    updateUserPersona,
    deleteUserPersona
  } = usePersonaStore();

  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [testingOpenAI, setTestingOpenAI] = useState(false);
  const [testingAnthropic, setTestingAnthropic] = useState(false);
  const [openAITestResult, setOpenAITestResult] = useState<{ valid: boolean; message: string } | null>(null);
  const [anthropicTestResult, setAnthropicTestResult] = useState<{ valid: boolean; message: string } | null>(null);

  // Persona state
  const [showPersonaForm, setShowPersonaForm] = useState(false);
  const [editingPersona, setEditingPersona] = useState<UserPersona | null>(null);
  const [personaForm, setPersonaForm] = useState<UserPersonaCreate>({
    name: '', role_title: '', organization: '', communication_style: '', negotiation_strengths: '', notes: '', is_default: false, preferred_provider: null, preferred_model: null
  });

  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
    email: '',
    openai_api_key: '',
    anthropic_api_key: '',
    preferred_provider: null as string | null,
    preferred_model: null as string | null,
  });

  useEffect(() => {
    if (user) {
      setFormData({
        first_name: user.first_name || '',
        last_name: user.last_name || '',
        email: user.email || '',
        openai_api_key: '',
        anthropic_api_key: '',
        preferred_provider: (user as any).preferred_provider || null,
        preferred_model: (user as any).preferred_model || null,
      });
      fetchUserPersonas(user.id);
    }
  }, [user]);

  // Persona handlers
  const handleEditPersona = (persona: UserPersona) => {
    setEditingPersona(persona);
    setPersonaForm({
      name: persona.name,
      role_title: persona.role_title || '',
      organization: persona.organization || '',
      communication_style: persona.communication_style || '',
      negotiation_strengths: persona.negotiation_strengths || '',
      notes: persona.notes || '',
      is_default: persona.is_default,
      preferred_provider: persona.preferred_provider || null,
      preferred_model: persona.preferred_model || null
    });
    setShowPersonaForm(true);
  };

  const handleSavePersona = async () => {
    if (!user) return;
    try {
      if (editingPersona) {
        await updateUserPersona(user.id, editingPersona.id, personaForm);
      } else {
        await createUserPersona(user.id, personaForm);
      }
      setShowPersonaForm(false);
      setEditingPersona(null);
      setPersonaForm({ name: '', role_title: '', organization: '', communication_style: '', negotiation_strengths: '', notes: '', is_default: false, preferred_provider: null, preferred_model: null });
    } catch (err) {
      console.error('Failed to save persona:', err);
    }
  };

  const handleDeletePersona = async (id: string) => {
    if (!user) return;
    if (confirm('Are you sure you want to delete this persona?')) {
      await deleteUserPersona(user.id, id);
    }
  };

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

      // Include provider preferences
      updateData.preferred_provider = formData.preferred_provider;
      updateData.preferred_model = formData.preferred_model;

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
    <div className="min-h-screen bg-chat-muted py-10 px-5">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/app')}
            className="text-chat-primary text-sm hover:underline mb-4 inline-flex"
          >
            ← Back to Chat
          </button>
          <h1 className="text-3xl font-semibold text-chat-foreground mb-2">
            User Profile
          </h1>
          <p className="text-base text-chat-muted-foreground">
            Manage your account settings and API keys
          </p>
        </div>

        {/* Profile Card */}
        <div className="bg-chat-card border border-chat-border rounded-lg p-8 shadow-card">
          {/* Account Information */}
          <div className="mb-8">
            <h2 className="text-lg font-semibold text-chat-foreground mb-4">
              Account Information
            </h2>
            <div className="grid [grid-template-columns:150px_1fr] gap-3 text-sm mb-5">
              <div className="text-chat-muted-foreground">Username:</div>
              <div className="text-chat-foreground font-medium">{user.username}</div>

              <div className="text-chat-muted-foreground">Role:</div>
              <div className="text-chat-foreground font-medium">{user.role}</div>
            </div>

            {/* API Keys Status */}
            <div className="mt-5">
              <h3 className="text-sm font-semibold text-chat-foreground mb-3">
                API Keys Status
              </h3>
              <div className="flex gap-4 flex-wrap">
                <div className={
                  'inline-flex items-center gap-2 px-3 py-2 rounded-md border text-sm ' +
                  (user.has_openai_key ? 'border-success/30 bg-success/10' : 'border-chat-border bg-transparent')
                }>
                  <span className={user.has_openai_key ? 'h-2 w-2 rounded-full bg-success inline-block' : 'h-2 w-2 rounded-full bg-chat-muted-foreground inline-block'} />
                  <span className={user.has_openai_key ? 'font-medium text-success' : 'font-medium text-chat-muted-foreground'}>OpenAI</span>
                </div>

                <div className={
                  'inline-flex items-center gap-2 px-3 py-2 rounded-md border text-sm ' +
                  (user.has_anthropic_key ? 'border-success/30 bg-success/10' : 'border-chat-border bg-transparent')
                }>
                  <span className={user.has_anthropic_key ? 'h-2 w-2 rounded-full bg-success inline-block' : 'h-2 w-2 rounded-full bg-chat-muted-foreground inline-block'} />
                  <span className={user.has_anthropic_key ? 'font-medium text-success' : 'font-medium text-chat-muted-foreground'}>Anthropic</span>
                </div>
              </div>
            </div>
          </div>

          {/* Edit Form */}
          <form onSubmit={handleSubmit}>
            <div className="mb-8">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-chat-foreground m-0">
                  Profile Details
                </h2>
                {!isEditing && (
                  <Button type="button" size="sm" onClick={() => setIsEditing(true)}>
                    Edit Profile
                  </Button>
                )}
              </div>

              <div className="flex flex-col gap-4">
                <div>
                  <label className="block text-sm font-medium text-chat-foreground mb-1.5">First Name</label>
                  <Input
                    type="text"
                    value={formData.first_name}
                    onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
                    disabled={!isEditing}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-chat-foreground mb-1.5">Last Name</label>
                  <Input
                    type="text"
                    value={formData.last_name}
                    onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
                    disabled={!isEditing}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-chat-foreground mb-1.5">Email</label>
                  <Input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    disabled={!isEditing}
                    required
                  />
                </div>
              </div>
            </div>

            {/* API Keys Section */}
            {isEditing && (
              <div className="mb-6">
                <h2 className="text-lg font-semibold text-chat-foreground mb-2">
                  API Keys
                </h2>
                <p className="text-xs text-chat-muted-foreground mb-4">
                  Leave blank to keep existing keys unchanged
                </p>

                <div className="flex flex-col gap-4">
                  {/* OpenAI API Key */}
                  <div>
                    <label className="block text-sm font-medium text-chat-foreground mb-1.5">
                      OpenAI API Key
                    </label>
                    <div className="flex gap-2">
                      <Input
                        type="password"
                        value={formData.openai_api_key}
                        onChange={(e) => {
                          setFormData({ ...formData, openai_api_key: e.target.value });
                          setOpenAITestResult(null);
                        }}
                        placeholder="sk-..."
                        className="flex-1"
                      />
                      <Button type="button" onClick={() => handleTestAPIKey('openai')} disabled={testingOpenAI || !formData.openai_api_key}>
                        {testingOpenAI ? 'Testing...' : 'Test Key'}
                      </Button>
                    </div>
                    {openAITestResult && (
                      <div className={openAITestResult.valid
                        ? 'mt-2 rounded-md border border-success/30 bg-success/10 px-3 py-2 text-xs text-success'
                        : 'mt-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger'}>
                        {openAITestResult.message}
                      </div>
                    )}
                  </div>

                  {/* Anthropic API Key */}
                  <div>
                    <label className="block text-sm font-medium text-chat-foreground mb-1.5">
                      Anthropic API Key
                    </label>
                    <div className="flex gap-2">
                      <Input
                        type="password"
                        value={formData.anthropic_api_key}
                        onChange={(e) => {
                          setFormData({ ...formData, anthropic_api_key: e.target.value });
                          setAnthropicTestResult(null);
                        }}
                        placeholder="sk-ant-..."
                        className="flex-1"
                      />
                      <Button type="button" onClick={() => handleTestAPIKey('anthropic')} disabled={testingAnthropic || !formData.anthropic_api_key}>
                        {testingAnthropic ? 'Testing...' : 'Test Key'}
                      </Button>
                    </div>
                    {anthropicTestResult && (
                      <div className={anthropicTestResult.valid
                        ? 'mt-2 rounded-md border border-success/30 bg-success/10 px-3 py-2 text-xs text-success'
                        : 'mt-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger'}>
                        {anthropicTestResult.message}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Default Provider Preferences Section */}
            {isEditing && (
              <div className="mb-6">
                <h2 className="text-lg font-semibold text-chat-foreground mb-2">
                  Default AI Provider
                </h2>
                <p className="text-xs text-chat-muted-foreground mb-4">
                  Set your preferred AI provider and model. This will be used as the default for all your negotiations.
                </p>

                <ProviderSelector
                  selectedProvider={formData.preferred_provider}
                  selectedModel={formData.preferred_model}
                  onProviderChange={(provider) => setFormData({ ...formData, preferred_provider: provider })}
                  onModelChange={(model) => setFormData({ ...formData, preferred_model: model })}
                  providerLabel="Default Provider"
                  modelLabel="Default Model"
                  showUseDefault={true}
                />
              </div>
            )}

            {/* Messages */}
            {error && (
              <div className="mb-4 rounded-md border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
                {error}
              </div>
            )}

            {success && (
              <div className="mb-4 rounded-md border border-success/30 bg-success/10 px-4 py-3 text-sm text-success">
                {success}
              </div>
            )}

            {/* Action Buttons */}
            {isEditing && (
              <div className="flex gap-3 justify-end">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    setIsEditing(false);
                    setError('');
                    setFormData({
                      first_name: user.first_name || '',
                      last_name: user.last_name || '',
                      email: user.email || '',
                      openai_api_key: '',
                      anthropic_api_key: '',
                      preferred_provider: (user as any).preferred_provider || null,
                      preferred_model: (user as any).preferred_model || null,
                    });
                  }}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={isSaving} variant="primary">
                  {isSaving ? 'Saving...' : 'Save Changes'}
                </Button>
              </div>
            )}
          </form>
        </div>

        {/* My Personas Section */}
        <div className="bg-chat-card border border-chat-border rounded-lg p-8 shadow-card mt-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold text-chat-foreground mb-1">
                My Personas
              </h2>
              <p className="text-xs text-chat-muted-foreground m-0">
                Your negotiation identities - the roles you play
              </p>
            </div>
            <Button
              onClick={() => {
                setEditingPersona(null);
                setPersonaForm({ name: '', role_title: '', organization: '', communication_style: '', negotiation_strengths: '', notes: '', is_default: false, preferred_provider: null, preferred_model: null });
                setShowPersonaForm(true);
              }}
            >
              + Add Persona
            </Button>
          </div>

          {userPersonas.length === 0 ? (
            <div className="text-center p-8 text-chat-muted-foreground">
              <p className="m-0">No personas yet. Create one to get started.</p>
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              {userPersonas.map((persona) => (
                <div
                  key={persona.id}
                  className="border border-chat-border rounded-lg p-4 flex justify-between items-start"
                >
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-base font-semibold text-chat-foreground">
                        {persona.name}
                      </span>
                      {persona.is_default && (
                        <Badge variant="success" className="text-[10px] py-0.5 px-2">Default</Badge>
                      )}
                    </div>
                    {persona.role_title && (
                      <p className="text-sm text-chat-foreground/70 mb-1">
                        {persona.role_title}{persona.organization ? ` at ${persona.organization}` : ''}
                      </p>
                    )}
                    {persona.communication_style && (
                      <p className="text-xs text-chat-muted-foreground m-0">
                        {persona.communication_style}
                      </p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" onClick={() => handleEditPersona(persona)}>
                      Edit
                    </Button>
                    <Button size="sm" variant="outline" className="text-danger border-danger/40" onClick={() => handleDeletePersona(persona.id)}>
                      Delete
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Persona Form Modal */}
        {showPersonaForm && (
          <div className="fixed inset-0 bg-black/50 z-[9999] flex items-center justify-center">
            <div className="bg-chat-card border border-chat-border rounded-xl w-[90%] max-w-[500px] max-h-[90vh] overflow-auto shadow-card">
              <div className="px-6 py-4 border-b border-chat-border flex items-center justify-between">
                <h3 className="m-0 text-lg font-semibold">
                  {editingPersona ? 'Edit Persona' : 'Create Persona'}
                </h3>
                <button
                  onClick={() => { setShowPersonaForm(false); setEditingPersona(null); }}
                  className="text-chat-muted-foreground text-2xl leading-none"
                >
                  ×
                </button>
              </div>
              <div className="p-6">
                <div className="mb-4">
                  <label className="block text-sm font-medium mb-1.5">Persona Name *</label>
                  <Input
                    type="text"
                    value={personaForm.name}
                    onChange={(e) => setPersonaForm({ ...personaForm, name: e.target.value })}
                    placeholder="e.g., Sales Director"
                  />
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium mb-1.5">Role / Title</label>
                  <Input
                    type="text"
                    value={personaForm.role_title || ''}
                    onChange={(e) => setPersonaForm({ ...personaForm, role_title: e.target.value })}
                    placeholder="e.g., VP of Sales"
                  />
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium mb-1.5">Organization</label>
                  <Input
                    type="text"
                    value={personaForm.organization || ''}
                    onChange={(e) => setPersonaForm({ ...personaForm, organization: e.target.value })}
                    placeholder="e.g., Acme Corp"
                  />
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium mb-1.5">Communication Style</label>
                  <Textarea
                    value={personaForm.communication_style || ''}
                    onChange={(e) => setPersonaForm({ ...personaForm, communication_style: e.target.value })}
                    placeholder="How do you prefer to communicate?"
                  />
                </div>
                <div className="mb-5">
                  <label className="block text-sm font-medium mb-1.5">Negotiation Strengths</label>
                  <Textarea
                    value={personaForm.negotiation_strengths || ''}
                    onChange={(e) => setPersonaForm({ ...personaForm, negotiation_strengths: e.target.value })}
                    placeholder="What are your key strengths?"
                  />
                </div>
                <div className="mb-5">
                  <label className="block text-sm font-medium mb-2">
                    AI Provider <span className="text-chat-muted-foreground font-normal">(optional)</span>
                  </label>
                  <p className="text-xs text-chat-muted-foreground mb-3">
                    Override the default AI provider when using this persona.
                  </p>
                  <ProviderSelector
                    selectedProvider={personaForm.preferred_provider || null}
                    selectedModel={personaForm.preferred_model || null}
                    onProviderChange={(provider) => setPersonaForm({ ...personaForm, preferred_provider: provider })}
                    onModelChange={(model) => setPersonaForm({ ...personaForm, preferred_model: model })}
                    providerLabel=""
                    modelLabel=""
                    showUseDefault={true}
                    compact={true}
                  />
                </div>
                <div className="mb-5">
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <input
                      type="checkbox"
                      checked={personaForm.is_default}
                      onChange={(e) => setPersonaForm({ ...personaForm, is_default: e.target.checked })}
                    />
                    Set as default persona
                  </label>
                </div>
                <div className="flex gap-3 justify-end">
                  <Button variant="outline" onClick={() => { setShowPersonaForm(false); setEditingPersona(null); }}>
                    Cancel
                  </Button>
                  <Button onClick={handleSavePersona} disabled={!personaForm.name}>
                    {editingPersona ? 'Save Changes' : 'Create Persona'}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
