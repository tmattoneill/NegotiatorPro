/**
 * PersonaManager - Manage User and Partner Personas (Tailwind)
 */
import { useState, useEffect } from 'react';
import { meaningfulLen, MIN_PERSONA_CHARS } from '../utils/personaContext';
import { useNavigate } from 'react-router-dom';
import { usePersonaStore } from '../store/personaStore';
import { useAuthStore } from '../store/authStore';
import type { UserPersona, PartnerPersona, UserPersonaCreate, PartnerPersonaCreate } from '../types/personas';
import Badge from './ui/Badge';
import Button from './ui/Button';
import { FaArrowLeft, FaRegUser, FaUsers, FaPlus, FaUserSlash, FaPen, FaTrash } from 'react-icons/fa';
import ProviderSelector from './ProviderSelector';

type TabType = 'user' | 'partner';

export default function PersonaManager() {
  const navigate = useNavigate();
  const { isAuthenticated, user } = useAuthStore();
  const {
    userPersonas, partnerPersonas,
    userPersonasLoading, partnerPersonasLoading,
    fetchUserPersonas, fetchPartnerPersonas,
    createUserPersona, createPartnerPersona,
    updateUserPersona, updatePartnerPersona,
    deleteUserPersona, deletePartnerPersona,
    error, clearError
  } = usePersonaStore();

  const [activeTab, setActiveTab] = useState<TabType>('user');
  const [showForm, setShowForm] = useState(false);
  const [editingPersona, setEditingPersona] = useState<UserPersona | PartnerPersona | null>(null);

  const userId = user?.id || '';

  useEffect(() => {
    if (!isAuthenticated || !userId) {
      navigate('/login');
      return;
    }
    fetchUserPersonas(userId);
    fetchPartnerPersonas(userId);
  }, [isAuthenticated, userId]);

  const handleCreateNew = () => {
    setEditingPersona(null);
    setShowForm(true);
  };

  // Removed unused handleEdit to satisfy strict TS

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this persona?')) return;
    if (activeTab === 'user') await deleteUserPersona(userId, id);
    else await deletePartnerPersona(userId, id);
  };

  const handleFormSubmit = async (data: UserPersonaCreate | PartnerPersonaCreate) => {
    if (activeTab === 'user') {
      if (editingPersona) await updateUserPersona(userId, editingPersona.id, data as Partial<UserPersonaCreate>);
      else await createUserPersona(userId, data as UserPersonaCreate);
    } else {
      if (editingPersona) await updatePartnerPersona(userId, editingPersona.id, data as Partial<PartnerPersonaCreate>);
      else await createPartnerPersona(userId, data as PartnerPersonaCreate);
    }
    setShowForm(false);
    setEditingPersona(null);
  };

  const isLoading = activeTab === 'user' ? userPersonasLoading : partnerPersonasLoading;
  const personas = activeTab === 'user' ? userPersonas : partnerPersonas;

  return (
    <div className="min-h-screen bg-chat-muted p-6">
      <div className="max-w-3xl mx-auto mb-4">
        <button onClick={() => navigate('/app')} className="text-chat-primary text-sm mb-4 inline-flex items-center gap-2">
          <FaArrowLeft /> Back
        </button>
        <h1 className="text-2xl font-semibold text-chat-foreground mb-1">Persona Manager</h1>
        <p className="text-sm text-chat-muted-foreground">Manage your negotiation identities and partner profiles</p>
      </div>

      <div className="max-w-3xl mx-auto flex gap-2 mb-2">
        <button
          className={`px-4 py-2 -mb-[2px] border-b-4 inline-flex items-center gap-1 ${activeTab === 'user' ? 'border-chat-primary text-chat-primary font-semibold bg-chat-primary/10' : 'border-transparent text-chat-muted-foreground hover:bg-chat-muted hover:text-chat-foreground'}`}
          onClick={() => setActiveTab('user')}
        >
          <FaRegUser /> My Personas
        </button>
        <button
          className={`px-4 py-2 -mb-[2px] border-b-4 inline-flex items-center gap-1 ${activeTab === 'partner' ? 'border-chat-primary text-chat-primary font-semibold bg-chat-primary/10' : 'border-transparent text-chat-muted-foreground hover:bg-chat-muted hover:text-chat-foreground'}`}
          onClick={() => setActiveTab('partner')}
        >
          <FaUsers /> Partner Personas
        </button>
      </div>

      <div className="max-w-3xl mx-auto bg-chat-card px-5 py-4 border-b border-chat-border rounded-t">
        <p className="text-sm text-chat-muted-foreground">
          {activeTab === 'user'
            ? 'Your negotiation identities are private to your account.'
            : 'Partner profiles you negotiate with. Mark as shared to make available to all users.'}
        </p>
      </div>

      {error && (
        <div className="max-w-3xl mx-auto mt-2 px-4 py-3 rounded bg-danger/10 border border-danger/30 text-danger flex items-center justify-between">
          <span>{error}</span>
          <button onClick={clearError} className="text-danger text-lg">×</button>
        </div>
      )}

      <div className="max-w-3xl mx-auto bg-chat-card px-5 py-4 flex justify-end border-x border-chat-border">
        <Button onClick={handleCreateNew} size="md" variant="primary">
          <FaPlus className="mr-2" />
          Create {activeTab === 'user' ? 'User' : 'Partner'} Persona
        </Button>
      </div>

      <div className="max-w-3xl mx-auto bg-chat-card px-5 py-4 border border-chat-border border-t-0 rounded-b min-h-[300px]">
        {isLoading ? (
          <div className="text-center py-12 text-chat-muted-foreground">Loading...</div>
        ) : personas.length === 0 ? (
          <div className="text-center py-12 text-chat-muted-foreground">
            <FaUserSlash className="text-4xl mb-3 opacity-50 mx-auto" />
            <p>No {activeTab === 'user' ? 'user' : 'partner'} personas yet.</p>
            <p className="text-sm opacity-70">Click "Create" to add your first one.</p>
          </div>
        ) : (
          personas.map((p) => (
            <PersonaCard key={p.id} persona={p} type={activeTab} onEdit={() => setEditingPersona(p)} onDelete={() => handleDelete(p.id)} />
          ))
        )}
      </div>

      {showForm && (
        <PersonaForm
          type={activeTab}
          persona={editingPersona}
          onSubmit={handleFormSubmit}
          onCancel={() => { setShowForm(false); setEditingPersona(null); }}
        />
      )}
    </div>
  );
}

interface PersonaCardProps {
  persona: UserPersona | PartnerPersona;
  type: TabType;
  onEdit: () => void;
  onDelete: () => void;
}

function PersonaCard({ persona, type, onEdit, onDelete }: PersonaCardProps) {
  const isUserPersona = type === 'user';
  const userPersona = persona as UserPersona;
  const partnerPersona = persona as PartnerPersona;

  return (
    <div className="border border-chat-border rounded-lg p-5 mb-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-lg font-semibold text-chat-foreground m-0">{persona.name}</h3>
          <p className="text-sm text-chat-muted-foreground mt-1">
            {isUserPersona
              ? `${userPersona.role_title || 'No title'} ${userPersona.organization ? `at ${userPersona.organization}` : ''}`
              : `${partnerPersona.role_title || 'No title'} ${partnerPersona.company ? `at ${partnerPersona.company}` : ''}`}
          </p>
        </div>
        <div className="flex gap-2">
          {isUserPersona && userPersona.is_default && <Badge variant="success">Default</Badge>}
          {!isUserPersona && partnerPersona.is_shared && <Badge variant="success">Shared</Badge>}
        </div>
      </div>

      {persona.communication_style && (
        <div className="mb-3">
          <label className="text-sm font-semibold text-chat-muted-foreground">Communication Style</label>
          <p className="text-sm text-chat-foreground">{persona.communication_style}</p>
        </div>
      )}

      {isUserPersona && userPersona.negotiation_strengths && (
        <div className="mb-3">
          <label className="text-sm font-semibold text-chat-muted-foreground">Strengths</label>
          <p className="text-sm text-chat-foreground">{userPersona.negotiation_strengths}</p>
        </div>
      )}

      {!isUserPersona && partnerPersona.known_interests && (
        <div className="mb-3">
          <label className="text-sm font-semibold text-chat-muted-foreground">Known Interests</label>
          <p className="text-sm text-chat-foreground">{partnerPersona.known_interests}</p>
        </div>
      )}

      {!isUserPersona && partnerPersona.batna_estimate && (
        <div className="mb-3">
          <label className="text-sm font-semibold text-chat-muted-foreground">BATNA Estimate</label>
          <p className="text-sm text-chat-foreground">{partnerPersona.batna_estimate}</p>
        </div>
      )}

      <div className="flex gap-2 mt-4 pt-4 border-t border-chat-border">
        <Button variant="outline" size="sm" onClick={onEdit}><FaPen className="mr-1" /> Edit</Button>
        <Button variant="danger" size="sm" onClick={onDelete}><FaTrash className="mr-1" /> Delete</Button>
      </div>
    </div>
  );
}

interface PersonaFormProps {
  type: TabType;
  persona: UserPersona | PartnerPersona | null;
  onSubmit: (data: UserPersonaCreate | PartnerPersonaCreate) => void;
  onCancel: () => void;
}

function PersonaForm({ type, persona, onSubmit, onCancel }: PersonaFormProps) {
  const isUserPersona = type === 'user';
  const [formData, setFormData] = useState(() => {
    if (persona) return { ...persona };
    return isUserPersona
      ? { name: '', role_title: '', organization: '', communication_style: '', negotiation_strengths: '', notes: '', is_default: false, preferred_provider: null as string | null, preferred_model: null as string | null }
      : { name: '', role_title: '', company: '', communication_style: '', known_interests: '', batna_estimate: '', relationship_notes: '', is_shared: false };
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData as UserPersonaCreate | PartnerPersonaCreate);
  };

  const updateField = (field: string, value: string | boolean | null) => setFormData((prev: any) => ({ ...prev, [field]: value }));

  const f = formData as any;
  const contextLen = isUserPersona
    ? meaningfulLen(f.role_title, f.organization, f.communication_style, f.negotiation_strengths, f.notes)
    : meaningfulLen(f.role_title, f.company, f.communication_style, f.known_interests, f.batna_estimate, f.relationship_notes);
  const meetsMin = contextLen >= MIN_PERSONA_CHARS;

  return (
    <div className="fixed inset-0 bg-black/50 z-[1000] flex items-center justify-center">
      <div className="bg-chat-card border border-chat-border rounded-xl w-[90%] max-w-[600px] max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between px-6 py-5 border-b border-chat-border">
          <h2 className="text-lg font-semibold">{persona ? 'Edit' : 'Create'} {isUserPersona ? 'User' : 'Partner'} Persona</h2>
          <button onClick={onCancel} className="text-2xl text-gray-600">×</button>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          <div>
            <label className="block text-[14px] font-medium">Name *</label>
            <input
              type="text"
              value={(formData as any).name}
              onChange={(e) => updateField('name', e.target.value)}
              required
              placeholder={isUserPersona ? 'e.g., Sales Director' : 'e.g., John Smith'}
              className="w-full px-3 py-2 border border-chat-border rounded text-[14px] mt-1 outline-none focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10"
            />
          </div>
          <div>
            <label className="block text-[14px] font-medium">Role / Title</label>
            <input
              type="text"
              value={(formData as any).role_title || ''}
              onChange={(e) => updateField('role_title', e.target.value)}
              placeholder="e.g., VP of Procurement"
              className="w-full px-3 py-2 border border-chat-border rounded text-[14px] mt-1 outline-none focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10"
            />
          </div>
          <div>
            <label className="block text-[14px] font-medium">{isUserPersona ? 'Organization' : 'Company'}</label>
            <input
              type="text"
              value={(isUserPersona ? (formData as any).organization : (formData as any).company) || ''}
              onChange={(e) => updateField(isUserPersona ? 'organization' : 'company', e.target.value)}
              placeholder="e.g., Acme Corp"
              className="w-full px-3 py-2 border border-chat-border rounded text-[14px] mt-1 outline-none focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10"
            />
          </div>
          <div>
            <label className="block text-[14px] font-medium">Communication Style</label>
            <textarea
              value={(formData as any).communication_style || ''}
              onChange={(e) => updateField('communication_style', e.target.value)}
              placeholder={isUserPersona ? 'How do you prefer to communicate in negotiations?' : 'How does this person typically communicate?'}
              className="w-full px-3 py-2 border border-chat-border rounded text-[14px] mt-1 outline-none focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 min-h-[80px] resize-y"
            />
          </div>
          {isUserPersona ? (
            <>
              <div>
                <label className="block text-[14px] font-medium">Negotiation Strengths</label>
                <textarea
                  value={(formData as any).negotiation_strengths || ''}
                  onChange={(e) => updateField('negotiation_strengths', e.target.value)}
                  placeholder="What are your key strengths as a negotiator?"
                  className="w-full px-3 py-2 border border-chat-border rounded text-[14px] mt-1 outline-none focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 min-h-[80px] resize-y"
                />
              </div>
              <div>
                <label className="block text-[14px] font-medium mb-2">
                  AI Provider <span className="text-chat-muted-foreground font-normal">(optional)</span>
                </label>
                <p className="text-xs text-chat-muted-foreground mb-3">
                  Override the default AI provider when using this persona.
                </p>
                <ProviderSelector
                  selectedProvider={(formData as any).preferred_provider || null}
                  selectedModel={(formData as any).preferred_model || null}
                  onProviderChange={(provider) => updateField('preferred_provider', provider)}
                  onModelChange={(model) => updateField('preferred_model', model)}
                  providerLabel=""
                  modelLabel=""
                  showUseDefault={true}
                  compact={true}
                />
              </div>
            </>
          ) : (
            <>
              <div>
                <label className="block text-[14px] font-medium">Known Interests</label>
                <textarea
                  value={(formData as any).known_interests || ''}
                  onChange={(e) => updateField('known_interests', e.target.value)}
                  placeholder="What are their priorities and interests?"
                  className="w-full px-3 py-2 border border-chat-border rounded text-[14px] mt-1 outline-none focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 min-h-[80px] resize-y"
                />
              </div>
              <div>
                <label className="block text-[14px] font-medium">BATNA Estimate</label>
                <textarea
                  value={(formData as any).batna_estimate || ''}
                  onChange={(e) => updateField('batna_estimate', e.target.value)}
                  placeholder="What's their Best Alternative to Negotiated Agreement?"
                  className="w-full px-3 py-2 border border-chat-border rounded text-[14px] mt-1 outline-none focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 min-h-[80px] resize-y"
                />
              </div>
            </>
          )}
          <div>
            <label className="inline-flex items-center gap-2 text-[14px] cursor-pointer">
              <input
                type="checkbox"
                checked={isUserPersona ? (formData as any).is_default : (formData as any).is_shared}
                onChange={(e) => updateField(isUserPersona ? 'is_default' : 'is_shared', e.target.checked)}
              />
              {isUserPersona ? ' Set as default persona' : ' Share with all users'}
            </label>
          </div>
          <div className="flex items-center justify-between gap-3 pt-2">
            <span className={`text-xs ${meetsMin ? 'text-chat-muted-foreground' : 'text-danger'}`}>
              {contextLen}/{MIN_PERSONA_CHARS} characters of context
              {!meetsMin && ` — add ${isUserPersona ? 'your role, style, strengths' : 'their role, interests, BATNA'}`}
            </span>
            <div className="flex gap-3">
              <Button variant="outline" type="button" onClick={onCancel}>Cancel</Button>
              <Button type="submit" disabled={!meetsMin}>{persona ? 'Save Changes' : 'Create Persona'}</Button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
