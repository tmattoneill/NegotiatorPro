/**
 * Modal for creating a new negotiation
 */
import { useState } from 'react';
import { useAuthStore } from '../store/authStore';
import { useNegotiationStore } from '../store/negotiationStore';
import { usePersonaStore } from '../store/personaStore';

interface NegotiationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function NegotiationModal({ isOpen, onClose }: NegotiationModalProps) {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const { user } = useAuthStore();
  const { createNegotiation } = useNegotiationStore();
  const { partnerPersonas } = usePersonaStore();

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
      // If no partner personas exist, create a default one
      let partnerIds: string[];

      if (partnerPersonas.length === 0) {
        // Create a default partner persona on the fly
        const { createPartnerPersona } = usePersonaStore.getState();
        const defaultPartner = await createPartnerPersona(user.id, {
          name: 'Partner',
          is_shared: false,
        });
        partnerIds = [defaultPartner.id];
      } else {
        partnerIds = [partnerPersonas[0].id];
      }

      const result = await createNegotiation(user.id, {
        title: title.trim(),
        description: description.trim() || undefined,
        partner_persona_ids: partnerIds,
      });

      if (result) {
        // Success - close modal and reset form
        setTitle('');
        setDescription('');
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
      setTitle('');
      setDescription('');
      setError('');
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 z-[1000] flex items-center justify-center" onClick={handleClose}>
      <div className="bg-white rounded-lg w-[90%] max-w-[500px] max-h-[90vh] overflow-auto shadow-lg" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-6 py-5 border-b border-border">
          <h2 className="text-xl font-semibold text-foreground m-0">Create New Negotiation</h2>
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
              <div className="px-3 py-2 mb-4 bg-red-50 border border-red-200 rounded text-[#c33]">{error}</div>
            )}

            <div className="mb-4">
              <label htmlFor="negotiation-title" className="block text-[14px] font-medium text-foreground">
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
                className="w-full px-3 py-2 border border-border rounded text-[14px] mt-2 outline-none focus:border-primary focus:ring-2 focus:ring-primary/10"
              />
            </div>

            <div className="mb-4">
              <label htmlFor="negotiation-description" className="block text-[14px] font-medium text-foreground">
                Description <span className="text-muted-foreground font-normal">(optional)</span>
              </label>
              <textarea
                id="negotiation-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of the negotiation context..."
                rows={4}
                disabled={isSubmitting}
                className="w-full px-3 py-2 border border-border rounded text-[14px] mt-2 outline-none focus:border-primary focus:ring-2 focus:ring-primary/10 resize-y"
              />
            </div>
          </div>

          <div className="px-6 py-4 border-t border-border flex justify-end">
            <button
              type="button"
              onClick={handleClose}
              disabled={isSubmitting}
              className="px-4 py-2 text-[14px] bg-white border border-border rounded mr-3 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !title.trim()}
              className="px-4 py-2 text-[14px] rounded bg-chat-primary text-white disabled:opacity-50"
            >
              {isSubmitting ? 'Creating...' : 'Create Negotiation'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
