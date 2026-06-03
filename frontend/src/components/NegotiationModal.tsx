/**
 * Modal for creating a new negotiation.
 *
 * Three-step flow:
 *   1. Negotiation basics (title, description, AI provider)
 *   2. Your profile — auto-link the default user persona if one exists,
 *      otherwise let the user pick an existing persona or create one inline
 *   3. Submit: creates the persona first (if needed) then the negotiation
 */
import { useEffect, useMemo, useState } from 'react';
import { useAuthStore } from '../store/authStore';
import { useNegotiationStore } from '../store/negotiationStore';
import { usePersonaStore } from '../store/personaStore';
import ProviderSelector from './ProviderSelector';
import Portal from './Portal';
import { meaningfulLen, MIN_PERSONA_CHARS } from '../utils/personaContext';

interface NegotiationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type PersonaMode = 'existing' | 'create' | 'skip';
type PartnerMode = 'existing' | 'create';

export default function NegotiationModal({ isOpen, onClose }: NegotiationModalProps) {
  // Basics
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  // Persona step
  const [personaMode, setPersonaMode] = useState<PersonaMode>('create');
  const [selectedPersonaId, setSelectedPersonaId] = useState<string>('');
  const [newPersonaName, setNewPersonaName] = useState('');
  const [newPersonaRole, setNewPersonaRole] = useState('');
  const [newPersonaOrg, setNewPersonaOrg] = useState('');
  const [newPersonaStyle, setNewPersonaStyle] = useState('');
  const [newPersonaStrengths, setNewPersonaStrengths] = useState('');
  const [newPersonaNotes, setNewPersonaNotes] = useState('');
  const [makeDefault, setMakeDefault] = useState(true);

  // Partner step
  const [partnerMode, setPartnerMode] = useState<PartnerMode>('create');
  const [selectedPartnerId, setSelectedPartnerId] = useState<string>('');
  const [newPartnerName, setNewPartnerName] = useState('');
  const [newPartnerRole, setNewPartnerRole] = useState('');
  const [newPartnerCompany, setNewPartnerCompany] = useState('');
  const [newPartnerStyle, setNewPartnerStyle] = useState('');
  const [newPartnerInterests, setNewPartnerInterests] = useState('');
  const [newPartnerBatna, setNewPartnerBatna] = useState('');
  const [newPartnerNotes, setNewPartnerNotes] = useState('');

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const { user } = useAuthStore();
  const { createNegotiation } = useNegotiationStore();
  const {
    partnerPersonas,
    userPersonas,
    fetchUserPersonas,
    fetchPartnerPersonas,
    createUserPersona,
  } = usePersonaStore();

  // The user's default persona (if any), found at the time the modal opens
  const defaultPersona = useMemo(
    () => userPersonas.find((p) => p.is_default) || null,
    [userPersonas],
  );

  // Fetch user personas when the modal opens. Pick a sensible initial mode:
  // - if user already has personas: default to "existing" with first selected
  // - otherwise: default to "create"
  useEffect(() => {
    if (!isOpen || !user?.id) return;
    fetchUserPersonas(user.id);
    fetchPartnerPersonas(user.id);
  }, [isOpen, user?.id, fetchUserPersonas, fetchPartnerPersonas]);

  useEffect(() => {
    if (userPersonas.length > 0 && !selectedPersonaId) {
      setPersonaMode('existing');
      setSelectedPersonaId(defaultPersona?.id ?? userPersonas[0].id);
    } else if (userPersonas.length === 0) {
      setPersonaMode('create');
    }
    // We deliberately don't reset state when the user has already interacted.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userPersonas.length, defaultPersona?.id]);

  useEffect(() => {
    if (partnerPersonas.length > 0 && !selectedPartnerId) {
      setPartnerMode('existing');
      setSelectedPartnerId(partnerPersonas[0].id);
    } else if (partnerPersonas.length === 0) {
      setPartnerMode('create');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [partnerPersonas.length]);

  // Context length gates (mirror the backend 240-char minimum). Only the
  // "create" path needs to meet it — existing/skip carry no new persona text.
  const userContextLen = meaningfulLen(
    newPersonaRole, newPersonaOrg, newPersonaStyle, newPersonaStrengths, newPersonaNotes,
  );
  const partnerContextLen = meaningfulLen(
    newPartnerRole, newPartnerCompany, newPartnerStyle, newPartnerInterests,
    newPartnerBatna, newPartnerNotes,
  );
  const userPersonaOk = personaMode !== 'create' || userContextLen >= MIN_PERSONA_CHARS;
  const partnerOk = partnerMode !== 'create' || partnerContextLen >= MIN_PERSONA_CHARS;
  const canSubmit = !!title.trim() && userPersonaOk && partnerOk && !isSubmitting;

  const resetForm = () => {
    setTitle('');
    setDescription('');
    setSelectedProvider(null);
    setSelectedModel(null);
    setPersonaMode(userPersonas.length > 0 ? 'existing' : 'create');
    setSelectedPersonaId(defaultPersona?.id ?? userPersonas[0]?.id ?? '');
    setNewPersonaName('');
    setNewPersonaRole('');
    setNewPersonaOrg('');
    setNewPersonaStyle('');
    setNewPersonaStrengths('');
    setNewPersonaNotes('');
    setMakeDefault(true);
    setPartnerMode(partnerPersonas.length > 0 ? 'existing' : 'create');
    setSelectedPartnerId(partnerPersonas[0]?.id ?? '');
    setNewPartnerName('');
    setNewPartnerRole('');
    setNewPartnerCompany('');
    setNewPartnerStyle('');
    setNewPartnerInterests('');
    setNewPartnerBatna('');
    setNewPartnerNotes('');
    setError('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!title.trim()) {
      setError('Negotiation name is required');
      return;
    }
    if (!user?.id) {
      setError('User not authenticated');
      return;
    }

    setIsSubmitting(true);

    try {
      // Step 1: resolve user persona ID from the picker
      let userPersonaId: string | undefined;

      if (personaMode === 'existing' && selectedPersonaId) {
        userPersonaId = selectedPersonaId;
      } else if (personaMode === 'create') {
        if (!newPersonaName.trim()) {
          setError('Profile name is required (or pick "Skip" to create without one)');
          setIsSubmitting(false);
          return;
        }
        const created = await createUserPersona(user.id, {
          name: newPersonaName.trim(),
          role_title: newPersonaRole.trim() || undefined,
          organization: newPersonaOrg.trim() || undefined,
          communication_style: newPersonaStyle.trim() || undefined,
          negotiation_strengths: newPersonaStrengths.trim() || undefined,
          notes: newPersonaNotes.trim() || undefined,
          is_default: makeDefault,
        });
        userPersonaId = created.id;
      } // personaMode === 'skip' leaves userPersonaId undefined

      // Step 2: resolve the negotiation partner from the picker
      const { createPartnerPersona } = usePersonaStore.getState();
      let partnerIds: string[];
      if (partnerMode === 'existing' && selectedPartnerId) {
        partnerIds = [selectedPartnerId];
      } else if (partnerMode === 'create') {
        if (!newPartnerName.trim()) {
          setError('Partner name is required');
          setIsSubmitting(false);
          return;
        }
        // No placeholder fallback any more — a partner needs real context
        // (>=240 chars) so the briefing can tailor strategy to them.
        const createdPartner = await createPartnerPersona(user.id, {
          name: newPartnerName.trim(),
          role_title: newPartnerRole.trim() || undefined,
          company: newPartnerCompany.trim() || undefined,
          communication_style: newPartnerStyle.trim() || undefined,
          known_interests: newPartnerInterests.trim() || undefined,
          batna_estimate: newPartnerBatna.trim() || undefined,
          relationship_notes: newPartnerNotes.trim() || undefined,
          is_shared: false,
        });
        partnerIds = [createdPartner.id];
      } else {
        setError('Choose or create a negotiation partner');
        setIsSubmitting(false);
        return;
      }

      // Step 3: create the negotiation
      const result = await createNegotiation(user.id, {
        title: title.trim(),
        description: description.trim() || undefined,
        user_persona_id: userPersonaId,
        partner_persona_ids: partnerIds,
        settings: selectedProvider
          ? { provider: selectedProvider, model: selectedModel }
          : undefined,
      });

      if (result) {
        resetForm();
        onClose();
      } else {
        setError('Failed to create negotiation');
      }
    } catch (err) {
      console.error('Error creating negotiation:', err);
      setError('An error occurred while creating the negotiation');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      resetForm();
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <Portal>
    <div className="fixed inset-0 bg-black/50 z-[1000] flex items-center justify-center" onClick={handleClose}>
      <div className="bg-chat-card text-chat-foreground border border-chat-border rounded-lg w-[90%] max-w-[560px] max-h-[90vh] overflow-auto shadow-lg" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-6 py-5 border-b border-chat-border">
          <h2 className="text-xl font-semibold text-chat-foreground m-0">Create New Negotiation</h2>
          <button
            className="text-[28px] text-gray-500 hover:bg-gray-100 w-8 h-8 rounded flex items-center justify-center disabled:opacity-50"
            onClick={handleClose}
            disabled={isSubmitting}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="px-6 py-5">
            {error && (
              <div className="px-3 py-2 mb-4 rounded border border-danger/30 bg-danger/10 text-danger">{error}</div>
            )}

            {/* Negotiation name */}
            <div className="mb-4">
              <label htmlFor="negotiation-title" className="block text-[14px] font-medium text-chat-foreground">
                Negotiation Name <span className="text-danger">*</span>
              </label>
              <input
                id="negotiation-title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g., Salary Negotiation, Vendor Contract"
                maxLength={255}
                disabled={isSubmitting}
                autoFocus
                className="w-full px-3 py-2 border border-chat-border rounded text-[14px] mt-2 outline-none focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10"
              />
            </div>

            {/* Description */}
            <div className="mb-4">
              <label htmlFor="negotiation-description" className="block text-[14px] font-medium text-chat-foreground">
                Description <span className="text-chat-muted-foreground font-normal">(optional)</span>
              </label>
              <textarea
                id="negotiation-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of the negotiation context: stakes, deadline, current state of play..."
                rows={4}
                disabled={isSubmitting}
                className="w-full px-3 py-2 border border-chat-border rounded text-[14px] mt-2 outline-none focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 resize-y"
              />
            </div>

            {/* Provider override */}
            <div className="mb-4">
              <label className="block text-[14px] font-medium text-chat-foreground mb-2">
                AI Provider <span className="text-chat-muted-foreground font-normal">(optional)</span>
              </label>
              <p className="text-xs text-chat-muted-foreground mb-3">
                Override the default provider for this negotiation, or leave as default.
              </p>
              <ProviderSelector
                selectedProvider={selectedProvider}
                selectedModel={selectedModel}
                onProviderChange={setSelectedProvider}
                onModelChange={setSelectedModel}
                providerLabel=""
                modelLabel=""
                showUseDefault={true}
                defaultOptionLabel="Use Profile Default"
                compact={true}
                disabled={isSubmitting}
              />
            </div>

            {/* Your profile — always selectable */}
            {(
              <div className="mb-2 pt-4 border-t border-chat-border">
                <label className="block text-[14px] font-medium text-chat-foreground mb-1">
                  Your Profile
                </label>
                <p className="text-xs text-chat-muted-foreground mb-3">
                  Tell the assistant who <em>you</em> are in this negotiation — role, style, strengths. Used in every chat to tailor advice.
                </p>

                {/* Mode selector */}
                <div className="flex flex-col gap-2 mb-3">
                  {userPersonas.length > 0 && (
                    <label className="flex items-center gap-2 text-[14px] text-chat-foreground">
                      <input
                        type="radio"
                        name="personaMode"
                        value="existing"
                        checked={personaMode === 'existing'}
                        onChange={() => setPersonaMode('existing')}
                        disabled={isSubmitting}
                      />
                      Use an existing profile
                    </label>
                  )}
                  <label className="flex items-center gap-2 text-[14px] text-chat-foreground">
                    <input
                      type="radio"
                      name="personaMode"
                      value="create"
                      checked={personaMode === 'create'}
                      onChange={() => setPersonaMode('create')}
                      disabled={isSubmitting}
                    />
                    Create a new profile
                  </label>
                  <label className="flex items-center gap-2 text-[14px] text-chat-foreground">
                    <input
                      type="radio"
                      name="personaMode"
                      value="skip"
                      checked={personaMode === 'skip'}
                      onChange={() => setPersonaMode('skip')}
                      disabled={isSubmitting}
                    />
                    Skip <span className="text-xs text-chat-muted-foreground">(advice won't be tailored to you)</span>
                  </label>
                </div>

                {/* Existing-persona dropdown */}
                {personaMode === 'existing' && userPersonas.length > 0 && (
                  <select
                    value={selectedPersonaId}
                    onChange={(e) => setSelectedPersonaId(e.target.value)}
                    disabled={isSubmitting}
                    className="w-full px-3 py-2 border border-chat-border rounded text-[14px] bg-chat-card"
                  >
                    {userPersonas.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}{p.role_title ? ` — ${p.role_title}` : ''}{p.is_default ? ' (default)' : ''}
                      </option>
                    ))}
                  </select>
                )}

                {/* Inline create form */}
                {personaMode === 'create' && (
                  <div className="grid grid-cols-1 gap-3">
                    <input
                      type="text"
                      value={newPersonaName}
                      onChange={(e) => setNewPersonaName(e.target.value)}
                      placeholder="Profile name (e.g. 'Matt — work negotiator')"
                      maxLength={255}
                      disabled={isSubmitting}
                      className="w-full px-3 py-2 border border-chat-border rounded text-[14px]"
                    />
                    <div className="grid grid-cols-2 gap-3">
                      <input
                        type="text"
                        value={newPersonaRole}
                        onChange={(e) => setNewPersonaRole(e.target.value)}
                        placeholder="Role / Title"
                        maxLength={255}
                        disabled={isSubmitting}
                        className="px-3 py-2 border border-chat-border rounded text-[14px]"
                      />
                      <input
                        type="text"
                        value={newPersonaOrg}
                        onChange={(e) => setNewPersonaOrg(e.target.value)}
                        placeholder="Organisation"
                        maxLength={255}
                        disabled={isSubmitting}
                        className="px-3 py-2 border border-chat-border rounded text-[14px]"
                      />
                    </div>
                    <textarea
                      value={newPersonaStyle}
                      onChange={(e) => setNewPersonaStyle(e.target.value)}
                      placeholder="Communication style (e.g. 'direct, evidence-led, prefers concise replies')"
                      rows={2}
                      disabled={isSubmitting}
                      className="px-3 py-2 border border-chat-border rounded text-[14px] resize-y"
                    />
                    <textarea
                      value={newPersonaStrengths}
                      onChange={(e) => setNewPersonaStrengths(e.target.value)}
                      placeholder="Negotiation strengths (e.g. 'anchoring, reframing, building rapport')"
                      rows={2}
                      disabled={isSubmitting}
                      className="px-3 py-2 border border-chat-border rounded text-[14px] resize-y"
                    />
                    <textarea
                      value={newPersonaNotes}
                      onChange={(e) => setNewPersonaNotes(e.target.value)}
                      placeholder="Other context the assistant should know about you (optional)"
                      rows={2}
                      disabled={isSubmitting}
                      className="px-3 py-2 border border-chat-border rounded text-[14px] resize-y"
                    />
                    <label className="flex items-center gap-2 text-[13px] text-chat-foreground">
                      <input
                        type="checkbox"
                        checked={makeDefault}
                        onChange={(e) => setMakeDefault(e.target.checked)}
                        disabled={isSubmitting}
                      />
                      Make this my default profile (auto-used for future negotiations)
                    </label>
                    <span className={`text-xs ${userContextLen >= MIN_PERSONA_CHARS ? 'text-chat-muted-foreground' : 'text-danger'}`}>
                      {userContextLen}/{MIN_PERSONA_CHARS} characters of context
                      {userContextLen < MIN_PERSONA_CHARS && ' — add your role, style and strengths'}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Negotiation partner — pick an existing one or create a new one */}
            <div className="mb-2 pt-4 border-t border-chat-border">
              <label className="block text-[14px] font-medium text-chat-foreground mb-1">
                Negotiation Partner
              </label>
              <p className="text-xs text-chat-muted-foreground mb-3">
                Who are you negotiating <em>with</em>? Used to tailor strategy and language.
              </p>

              <div className="flex flex-col gap-2 mb-3">
                {partnerPersonas.length > 0 && (
                  <label className="flex items-center gap-2 text-[14px] text-chat-foreground">
                    <input
                      type="radio"
                      name="partnerMode"
                      value="existing"
                      checked={partnerMode === 'existing'}
                      onChange={() => setPartnerMode('existing')}
                      disabled={isSubmitting}
                    />
                    Use an existing partner
                  </label>
                )}
                <label className="flex items-center gap-2 text-[14px] text-chat-foreground">
                  <input
                    type="radio"
                    name="partnerMode"
                    value="create"
                    checked={partnerMode === 'create'}
                    onChange={() => setPartnerMode('create')}
                    disabled={isSubmitting}
                  />
                  Create a new partner
                </label>
              </div>

              {partnerMode === 'existing' && partnerPersonas.length > 0 && (
                <select
                  value={selectedPartnerId}
                  onChange={(e) => setSelectedPartnerId(e.target.value)}
                  disabled={isSubmitting}
                  className="w-full px-3 py-2 border border-chat-border rounded text-[14px] bg-chat-card"
                >
                  {partnerPersonas.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}{p.company ? ` — ${p.company}` : (p.role_title ? ` — ${p.role_title}` : '')}
                    </option>
                  ))}
                </select>
              )}

              {partnerMode === 'create' && (
                <div className="grid grid-cols-1 gap-3">
                  <input
                    type="text"
                    value={newPartnerName}
                    onChange={(e) => setNewPartnerName(e.target.value)}
                    placeholder="Partner name (e.g. 'Acme Corp', 'Jane Doe')"
                    maxLength={255}
                    disabled={isSubmitting}
                    className="w-full px-3 py-2 border border-chat-border rounded text-[14px]"
                  />
                  <div className="grid grid-cols-2 gap-3">
                    <input
                      type="text"
                      value={newPartnerRole}
                      onChange={(e) => setNewPartnerRole(e.target.value)}
                      placeholder="Role / Title"
                      maxLength={255}
                      disabled={isSubmitting}
                      className="px-3 py-2 border border-chat-border rounded text-[14px]"
                    />
                    <input
                      type="text"
                      value={newPartnerCompany}
                      onChange={(e) => setNewPartnerCompany(e.target.value)}
                      placeholder="Company"
                      maxLength={255}
                      disabled={isSubmitting}
                      className="px-3 py-2 border border-chat-border rounded text-[14px]"
                    />
                  </div>
                  <textarea
                    value={newPartnerStyle}
                    onChange={(e) => setNewPartnerStyle(e.target.value)}
                    placeholder="Communication style (e.g. 'aggressive opener, anchors high, goes quiet under pressure')"
                    rows={2}
                    disabled={isSubmitting}
                    className="px-3 py-2 border border-chat-border rounded text-[14px] resize-y"
                  />
                  <textarea
                    value={newPartnerInterests}
                    onChange={(e) => setNewPartnerInterests(e.target.value)}
                    placeholder="Known interests / priorities (what do they actually want — price, speed, certainty, status?)"
                    rows={2}
                    disabled={isSubmitting}
                    className="px-3 py-2 border border-chat-border rounded text-[14px] resize-y"
                  />
                  <textarea
                    value={newPartnerBatna}
                    onChange={(e) => setNewPartnerBatna(e.target.value)}
                    placeholder="BATNA estimate (their best alternative if this deal falls through)"
                    rows={2}
                    disabled={isSubmitting}
                    className="px-3 py-2 border border-chat-border rounded text-[14px] resize-y"
                  />
                  <textarea
                    value={newPartnerNotes}
                    onChange={(e) => setNewPartnerNotes(e.target.value)}
                    placeholder="Relationship notes (history, leverage, constraints, anything else that matters)"
                    rows={2}
                    disabled={isSubmitting}
                    className="px-3 py-2 border border-chat-border rounded text-[14px] resize-y"
                  />
                  <span className={`text-xs ${partnerContextLen >= MIN_PERSONA_CHARS ? 'text-chat-muted-foreground' : 'text-danger'}`}>
                    {partnerContextLen}/{MIN_PERSONA_CHARS} characters of context
                    {partnerContextLen < MIN_PERSONA_CHARS && ' — add their style, interests and BATNA'}
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="px-6 py-4 border-t border-chat-border flex justify-end">
            <button
              type="button"
              onClick={handleClose}
              disabled={isSubmitting}
              className="px-4 py-2 text-[14px] bg-chat-card text-chat-foreground border border-chat-border rounded mr-3 hover:bg-chat-muted disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!canSubmit}
              className="px-4 py-2 text-[14px] rounded bg-chat-primary text-white disabled:opacity-50"
            >
              {isSubmitting ? 'Creating...' : 'Create Negotiation'}
            </button>
          </div>
        </form>
      </div>
    </div>
    </Portal>
  );
}
