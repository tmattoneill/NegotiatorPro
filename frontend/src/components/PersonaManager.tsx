/**
 * PersonaManager - Manage User and Partner Personas
 *
 * User Personas: Private to the user account
 * Partner Personas: Can be shared across users
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePersonaStore } from '../store/personaStore';
import { useAuthStore } from '../store/authStore';
import type { UserPersona, PartnerPersona, UserPersonaCreate, PartnerPersonaCreate } from '../types/personas';

type TabType = 'user' | 'partner';

export default function PersonaManager() {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuthStore();
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

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }
    fetchUserPersonas();
    fetchPartnerPersonas();
  }, [isAuthenticated]);

  const handleCreateNew = () => {
    setEditingPersona(null);
    setShowForm(true);
  };

  const handleEdit = (persona: UserPersona | PartnerPersona) => {
    setEditingPersona(persona);
    setShowForm(true);
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this persona?')) return;

    if (activeTab === 'user') {
      await deleteUserPersona(id);
    } else {
      await deletePartnerPersona(id);
    }
  };

  const handleFormSubmit = async (data: UserPersonaCreate | PartnerPersonaCreate) => {
    if (activeTab === 'user') {
      if (editingPersona) {
        await updateUserPersona(editingPersona.id, data as Partial<UserPersonaCreate>);
      } else {
        await createUserPersona(data as UserPersonaCreate);
      }
    } else {
      if (editingPersona) {
        await updatePartnerPersona(editingPersona.id, data as Partial<PartnerPersonaCreate>);
      } else {
        await createPartnerPersona(data as PartnerPersonaCreate);
      }
    }
    setShowForm(false);
    setEditingPersona(null);
  };

  const isLoading = activeTab === 'user' ? userPersonasLoading : partnerPersonasLoading;
  const personas = activeTab === 'user' ? userPersonas : partnerPersonas;

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <button onClick={() => navigate('/')} style={styles.backButton}>
          <i className="fa-light fa-arrow-left"></i> Back
        </button>
        <h1 style={styles.title}>Persona Manager</h1>
        <p style={styles.subtitle}>
          Manage your negotiation identities and partner profiles
        </p>
      </div>

      {/* Tabs */}
      <div style={styles.tabs}>
        <button
          style={{ ...styles.tab, ...(activeTab === 'user' ? styles.tabActive : {}) }}
          onClick={() => setActiveTab('user')}
        >
          <i className="fa-light fa-user"></i> My Personas
        </button>
        <button
          style={{ ...styles.tab, ...(activeTab === 'partner' ? styles.tabActive : {}) }}
          onClick={() => setActiveTab('partner')}
        >
          <i className="fa-light fa-users"></i> Partner Personas
        </button>
      </div>

      {/* Tab Description */}
      <div style={styles.tabDescription}>
        {activeTab === 'user' ? (
          <p>Your negotiation identities - the roles you play in negotiations. These are private to your account.</p>
        ) : (
          <p>Profiles of people you negotiate with. Mark as "shared" to make available to all users.</p>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div style={styles.error}>
          {error}
          <button onClick={clearError} style={styles.errorClose}>×</button>
        </div>
      )}

      {/* Action Bar */}
      <div style={styles.actionBar}>
        <button onClick={handleCreateNew} style={styles.createButton}>
          <i className="fa-light fa-plus"></i> Create {activeTab === 'user' ? 'User' : 'Partner'} Persona
        </button>
      </div>

      {/* Persona List */}
      <div style={styles.list}>
        {isLoading ? (
          <div style={styles.loading}>Loading...</div>
        ) : personas.length === 0 ? (
          <div style={styles.empty}>
            <i className="fa-light fa-user-slash" style={{ fontSize: '48px', marginBottom: '16px', opacity: 0.5 }}></i>
            <p>No {activeTab === 'user' ? 'user' : 'partner'} personas yet.</p>
            <p style={{ fontSize: '13px', opacity: 0.7 }}>
              Click "Create" to add your first one.
            </p>
          </div>
        ) : (
          personas.map((persona) => (
            <PersonaCard
              key={persona.id}
              persona={persona}
              type={activeTab}
              onEdit={() => handleEdit(persona)}
              onDelete={() => handleDelete(persona.id)}
            />
          ))
        )}
      </div>

      {/* Form Modal */}
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

// Persona Card Component
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
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <div>
          <h3 style={styles.cardTitle}>{persona.name}</h3>
          <p style={styles.cardSubtitle}>
            {isUserPersona
              ? `${userPersona.role_title || 'No title'} ${userPersona.organization ? `at ${userPersona.organization}` : ''}`
              : `${partnerPersona.role_title || 'No title'} ${partnerPersona.company ? `at ${partnerPersona.company}` : ''}`
            }
          </p>
        </div>
        <div style={styles.cardBadges}>
          {isUserPersona && userPersona.is_default && (
            <span style={styles.badgeDefault}>Default</span>
          )}
          {!isUserPersona && partnerPersona.is_shared && (
            <span style={styles.badgeShared}>Shared</span>
          )}
        </div>
      </div>

      {persona.communication_style && (
        <div style={styles.cardField}>
          <label>Communication Style</label>
          <p>{persona.communication_style}</p>
        </div>
      )}

      {isUserPersona && userPersona.negotiation_strengths && (
        <div style={styles.cardField}>
          <label>Strengths</label>
          <p>{userPersona.negotiation_strengths}</p>
        </div>
      )}

      {!isUserPersona && partnerPersona.known_interests && (
        <div style={styles.cardField}>
          <label>Known Interests</label>
          <p>{partnerPersona.known_interests}</p>
        </div>
      )}

      {!isUserPersona && partnerPersona.batna_estimate && (
        <div style={styles.cardField}>
          <label>BATNA Estimate</label>
          <p>{partnerPersona.batna_estimate}</p>
        </div>
      )}

      <div style={styles.cardActions}>
        <button onClick={onEdit} style={styles.editButton}>
          <i className="fa-light fa-pen"></i> Edit
        </button>
        <button onClick={onDelete} style={styles.deleteButton}>
          <i className="fa-light fa-trash"></i> Delete
        </button>
      </div>
    </div>
  );
}

// Persona Form Component
interface PersonaFormProps {
  type: TabType;
  persona: UserPersona | PartnerPersona | null;
  onSubmit: (data: UserPersonaCreate | PartnerPersonaCreate) => void;
  onCancel: () => void;
}

function PersonaForm({ type, persona, onSubmit, onCancel }: PersonaFormProps) {
  const isUserPersona = type === 'user';
  const [formData, setFormData] = useState(() => {
    if (persona) {
      return { ...persona };
    }
    return isUserPersona
      ? { name: '', role_title: '', organization: '', communication_style: '', negotiation_strengths: '', notes: '', is_default: false }
      : { name: '', role_title: '', company: '', communication_style: '', known_interests: '', batna_estimate: '', relationship_notes: '', is_shared: false };
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData as UserPersonaCreate | PartnerPersonaCreate);
  };

  const updateField = (field: string, value: string | boolean) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div style={styles.modalOverlay}>
      <div style={styles.modal}>
        <div style={styles.modalHeader}>
          <h2>{persona ? 'Edit' : 'Create'} {isUserPersona ? 'User' : 'Partner'} Persona</h2>
          <button onClick={onCancel} style={styles.modalClose}>×</button>
        </div>

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.formGroup}>
            <label>Name *</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => updateField('name', e.target.value)}
              required
              placeholder={isUserPersona ? "e.g., Sales Director" : "e.g., John Smith"}
              style={styles.input}
            />
          </div>

          <div style={styles.formGroup}>
            <label>Role / Title</label>
            <input
              type="text"
              value={formData.role_title || ''}
              onChange={(e) => updateField('role_title', e.target.value)}
              placeholder="e.g., VP of Procurement"
              style={styles.input}
            />
          </div>

          <div style={styles.formGroup}>
            <label>{isUserPersona ? 'Organization' : 'Company'}</label>
            <input
              type="text"
              value={(isUserPersona ? (formData as any).organization : (formData as any).company) || ''}
              onChange={(e) => updateField(isUserPersona ? 'organization' : 'company', e.target.value)}
              placeholder="e.g., Acme Corp"
              style={styles.input}
            />
          </div>

          <div style={styles.formGroup}>
            <label>Communication Style</label>
            <textarea
              value={formData.communication_style || ''}
              onChange={(e) => updateField('communication_style', e.target.value)}
              placeholder={isUserPersona
                ? "How do you prefer to communicate in negotiations?"
                : "How does this person typically communicate?"
              }
              style={styles.textarea}
            />
          </div>

          {isUserPersona ? (
            <div style={styles.formGroup}>
              <label>Negotiation Strengths</label>
              <textarea
                value={(formData as any).negotiation_strengths || ''}
                onChange={(e) => updateField('negotiation_strengths', e.target.value)}
                placeholder="What are your key strengths as a negotiator?"
                style={styles.textarea}
              />
            </div>
          ) : (
            <>
              <div style={styles.formGroup}>
                <label>Known Interests</label>
                <textarea
                  value={(formData as any).known_interests || ''}
                  onChange={(e) => updateField('known_interests', e.target.value)}
                  placeholder="What are their priorities and interests?"
                  style={styles.textarea}
                />
              </div>
              <div style={styles.formGroup}>
                <label>BATNA Estimate</label>
                <textarea
                  value={(formData as any).batna_estimate || ''}
                  onChange={(e) => updateField('batna_estimate', e.target.value)}
                  placeholder="What's their Best Alternative to Negotiated Agreement?"
                  style={styles.textarea}
                />
              </div>
            </>
          )}

          <div style={styles.formGroup}>
            <label>Notes</label>
            <textarea
              value={(isUserPersona ? (formData as any).notes : (formData as any).relationship_notes) || ''}
              onChange={(e) => updateField(isUserPersona ? 'notes' : 'relationship_notes', e.target.value)}
              placeholder="Any additional notes..."
              style={styles.textarea}
            />
          </div>

          <div style={styles.formCheckbox}>
            <label>
              <input
                type="checkbox"
                checked={isUserPersona ? (formData as any).is_default : (formData as any).is_shared}
                onChange={(e) => updateField(isUserPersona ? 'is_default' : 'is_shared', e.target.checked)}
              />
              {isUserPersona
                ? ' Set as default persona'
                : ' Share with all users'
              }
            </label>
          </div>

          <div style={styles.formActions}>
            <button type="button" onClick={onCancel} style={styles.cancelButton}>
              Cancel
            </button>
            <button type="submit" style={styles.submitButton}>
              {persona ? 'Save Changes' : 'Create Persona'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// Styles
const styles: Record<string, React.CSSProperties> = {
  container: {
    minHeight: '100vh',
    background: '#efefef',
    padding: '24px',
  },
  header: {
    maxWidth: '900px',
    margin: '0 auto 24px',
  },
  backButton: {
    background: 'none',
    border: 'none',
    color: '#3498db',
    fontSize: '14px',
    cursor: 'pointer',
    marginBottom: '16px',
    padding: 0,
  },
  title: {
    fontSize: '28px',
    fontWeight: 600,
    color: '#191919',
    margin: '0 0 8px',
  },
  subtitle: {
    fontSize: '14px',
    color: '#9fadbd',
    margin: 0,
  },
  tabs: {
    maxWidth: '900px',
    margin: '0 auto',
    display: 'flex',
    gap: '8px',
    marginBottom: '8px',
  },
  tab: {
    padding: '12px 24px',
    background: '#fff',
    border: '1px solid #ddd',
    borderBottom: 'none',
    borderRadius: '8px 8px 0 0',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    color: '#666',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  tabActive: {
    background: '#fff',
    color: '#191919',
    borderColor: '#3498db',
    borderBottomColor: '#fff',
  },
  tabDescription: {
    maxWidth: '900px',
    margin: '0 auto',
    background: '#fff',
    padding: '16px 20px',
    borderRadius: '0 8px 0 0',
    fontSize: '13px',
    color: '#666',
    borderBottom: '1px solid #eee',
  },
  actionBar: {
    maxWidth: '900px',
    margin: '0 auto',
    background: '#fff',
    padding: '16px 20px',
    display: 'flex',
    justifyContent: 'flex-end',
  },
  createButton: {
    padding: '10px 20px',
    background: '#3498db',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 500,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  list: {
    maxWidth: '900px',
    margin: '0 auto',
    background: '#fff',
    padding: '20px',
    borderRadius: '0 0 8px 8px',
    minHeight: '300px',
  },
  loading: {
    textAlign: 'center',
    padding: '48px',
    color: '#666',
  },
  empty: {
    textAlign: 'center',
    padding: '48px',
    color: '#666',
  },
  error: {
    maxWidth: '900px',
    margin: '0 auto 16px',
    background: '#fee',
    color: '#c00',
    padding: '12px 16px',
    borderRadius: '6px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  errorClose: {
    background: 'none',
    border: 'none',
    fontSize: '20px',
    cursor: 'pointer',
    color: '#c00',
  },
  card: {
    border: '1px solid #eee',
    borderRadius: '8px',
    padding: '20px',
    marginBottom: '16px',
  },
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '12px',
  },
  cardTitle: {
    fontSize: '18px',
    fontWeight: 600,
    color: '#191919',
    margin: 0,
  },
  cardSubtitle: {
    fontSize: '13px',
    color: '#666',
    margin: '4px 0 0',
  },
  cardBadges: {
    display: 'flex',
    gap: '8px',
  },
  badgeDefault: {
    background: '#3498db',
    color: '#fff',
    padding: '4px 10px',
    borderRadius: '12px',
    fontSize: '11px',
    fontWeight: 500,
  },
  badgeShared: {
    background: '#27ae60',
    color: '#fff',
    padding: '4px 10px',
    borderRadius: '12px',
    fontSize: '11px',
    fontWeight: 500,
  },
  cardField: {
    marginBottom: '12px',
  },
  cardActions: {
    display: 'flex',
    gap: '8px',
    marginTop: '16px',
    paddingTop: '16px',
    borderTop: '1px solid #eee',
  },
  editButton: {
    padding: '8px 16px',
    background: '#f5f5f5',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '13px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  deleteButton: {
    padding: '8px 16px',
    background: '#fff',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '13px',
    cursor: 'pointer',
    color: '#c00',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  modalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 9999,
  },
  modal: {
    background: '#fff',
    borderRadius: '12px',
    width: '90%',
    maxWidth: '600px',
    maxHeight: '90vh',
    overflow: 'auto',
  },
  modalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '20px 24px',
    borderBottom: '1px solid #eee',
  },
  modalClose: {
    background: 'none',
    border: 'none',
    fontSize: '24px',
    cursor: 'pointer',
    color: '#666',
  },
  form: {
    padding: '24px',
  },
  formGroup: {
    marginBottom: '20px',
  },
  input: {
    width: '100%',
    padding: '10px 12px',
    border: '1px solid #ddd',
    borderRadius: '6px',
    fontSize: '14px',
    marginTop: '6px',
    boxSizing: 'border-box',
  },
  textarea: {
    width: '100%',
    padding: '10px 12px',
    border: '1px solid #ddd',
    borderRadius: '6px',
    fontSize: '14px',
    marginTop: '6px',
    minHeight: '80px',
    resize: 'vertical',
    boxSizing: 'border-box',
  },
  formCheckbox: {
    marginBottom: '24px',
  },
  formActions: {
    display: 'flex',
    gap: '12px',
    justifyContent: 'flex-end',
  },
  cancelButton: {
    padding: '10px 20px',
    background: '#f5f5f5',
    border: '1px solid #ddd',
    borderRadius: '6px',
    fontSize: '14px',
    cursor: 'pointer',
  },
  submitButton: {
    padding: '10px 20px',
    background: '#3498db',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 500,
    cursor: 'pointer',
  },
};
