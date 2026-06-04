/**
 * PartnerPersonaModal
 *
 * Edit the partner bound to the active negotiation with copy-on-write: the first
 * edit clones a shared template into a private, negotiation-scoped copy
 * (PUT /api/negotiations/{id}/partner) so the template and other negotiations are
 * never touched. The backend enforces a minimum-context rule on partner edits, so
 * its 422 detail is surfaced inline rather than failing silently.
 */
import { useEffect, useState } from 'react';
import { useAuthStore } from '../store/authStore';
import { editNegotiationPartner } from '../services/api';
import type { PartnerPersona, PartnerPersonaUpdate } from '../types/personas';
import Button from './ui/Button';
import Input from './ui/Input';
import Textarea from './ui/Textarea';

interface PartnerPersonaModalProps {
  isOpen: boolean;
  partner: PartnerPersona | null;
  negotiationId: string | null;
  onClose: () => void;
  onSaved: () => void;
}

const emptyForm: PartnerPersonaUpdate = {
  name: '',
  role_title: '',
  company: '',
  communication_style: '',
  known_interests: '',
  batna_estimate: '',
  relationship_notes: '',
};

export default function PartnerPersonaModal({ isOpen, partner, negotiationId, onClose, onSaved }: PartnerPersonaModalProps) {
  const { user } = useAuthStore();
  const [form, setForm] = useState<PartnerPersonaUpdate>(emptyForm);
  const [updateParent, setUpdateParent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // Re-seed the form whenever a different partner is opened.
  useEffect(() => {
    if (isOpen && partner) {
      setForm({
        name: partner.name,
        role_title: partner.role_title || '',
        company: partner.company || '',
        communication_style: partner.communication_style || '',
        known_interests: partner.known_interests || '',
        batna_estimate: partner.batna_estimate || '',
        relationship_notes: partner.relationship_notes || '',
      });
      setUpdateParent(false);
      setError(null);
    }
  }, [isOpen, partner?.id]);

  if (!isOpen || !partner) return null;

  const handleSave = async () => {
    if (!user?.id || !negotiationId) return;
    setSaving(true);
    setError(null);
    try {
      await editNegotiationPartner(user.id, negotiationId, form, updateParent);
      onSaved();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save partner. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-[9999] flex items-center justify-center">
      <div className="bg-chat-card text-chat-foreground border border-chat-border rounded-xl w-[90%] max-w-[500px] max-h-[90vh] overflow-auto shadow-card">
        <div className="px-6 py-4 border-b border-chat-border flex items-center justify-between">
          <div>
            <h3 className="m-0 text-lg font-semibold">Edit Partner</h3>
            <p className="m-0 mt-0.5 text-xs text-chat-muted-foreground">Changes apply only to this negotiation</p>
          </div>
          <button onClick={onClose} className="text-chat-muted-foreground text-2xl leading-none">×</button>
        </div>
        <div className="p-6">
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1.5">Partner Name *</label>
            <Input
              type="text"
              value={form.name || ''}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="e.g., The Media Trust"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1.5">Role / Title</label>
            <Input
              type="text"
              value={form.role_title || ''}
              onChange={(e) => setForm({ ...form, role_title: e.target.value })}
              placeholder="e.g., Head of Procurement"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1.5">Company</label>
            <Input
              type="text"
              value={form.company || ''}
              onChange={(e) => setForm({ ...form, company: e.target.value })}
              placeholder="e.g., Acme Corp"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1.5">Communication Style</label>
            <Textarea
              value={form.communication_style || ''}
              onChange={(e) => setForm({ ...form, communication_style: e.target.value })}
              placeholder="How do they communicate?"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1.5">Known Interests</label>
            <Textarea
              value={form.known_interests || ''}
              onChange={(e) => setForm({ ...form, known_interests: e.target.value })}
              placeholder="What do they care about?"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1.5">BATNA Estimate</label>
            <Textarea
              value={form.batna_estimate || ''}
              onChange={(e) => setForm({ ...form, batna_estimate: e.target.value })}
              placeholder="Their best alternative to a negotiated agreement"
            />
          </div>
          <div className="mb-5">
            <label className="block text-sm font-medium mb-1.5">Relationship Notes</label>
            <Textarea
              value={form.relationship_notes || ''}
              onChange={(e) => setForm({ ...form, relationship_notes: e.target.value })}
              placeholder="History, rapport, prior dealings"
            />
          </div>
          <div className="mb-5">
            <label className="flex items-start gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={updateParent}
                onChange={(e) => setUpdateParent(e.target.checked)}
              />
              <span>
                Also update the shared template
                <span className="block text-xs text-chat-muted-foreground">
                  Applies this change to the original partner too, affecting other negotiations that use it.
                </span>
              </span>
            </label>
          </div>
          {error && (
            <div className="mb-4 rounded-md border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
              {error}
            </div>
          )}
          <div className="flex gap-3 justify-end">
            <Button variant="outline" onClick={onClose}>Cancel</Button>
            <Button onClick={handleSave} disabled={!form.name || saving}>
              {saving ? 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
