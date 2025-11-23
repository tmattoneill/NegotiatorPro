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

    // Find a partner persona - use first available or create error if none exist
    if (partnerPersonas.length === 0) {
      setError('No partner personas available. Please create a partner persona first from your profile.');
      return;
    }

    const partnerIds = [partnerPersonas[0].id];

    setIsSubmitting(true);

    try {
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
    <div className="modal-overlay" onClick={handleClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Create New Negotiation</h2>
          <button
            className="modal-close"
            onClick={handleClose}
            disabled={isSubmitting}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            {error && (
              <div className="error-message" style={{
                padding: '12px',
                marginBottom: '16px',
                background: '#fee',
                border: '1px solid #fcc',
                borderRadius: '4px',
                color: '#c33'
              }}>
                {error}
              </div>
            )}

            <div className="form-group">
              <label htmlFor="negotiation-title">
                Negotiation Name <span style={{ color: '#e74c3c' }}>*</span>
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
                style={{
                  width: '100%',
                  padding: '10px',
                  fontSize: '14px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  marginTop: '8px',
                }}
              />
            </div>

            <div className="form-group" style={{ marginTop: '20px' }}>
              <label htmlFor="negotiation-description">
                Description <span style={{ color: '#999', fontWeight: 'normal' }}>(optional)</span>
              </label>
              <textarea
                id="negotiation-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of the negotiation context..."
                rows={4}
                disabled={isSubmitting}
                style={{
                  width: '100%',
                  padding: '10px',
                  fontSize: '14px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  marginTop: '8px',
                  resize: 'vertical',
                  fontFamily: 'inherit',
                }}
              />
            </div>
          </div>

          <div className="modal-footer">
            <button
              type="button"
              onClick={handleClose}
              disabled={isSubmitting}
              style={{
                padding: '10px 20px',
                fontSize: '14px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                background: '#fff',
                cursor: isSubmitting ? 'not-allowed' : 'pointer',
                marginRight: '12px',
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !title.trim()}
              style={{
                padding: '10px 20px',
                fontSize: '14px',
                border: 'none',
                borderRadius: '4px',
                background: isSubmitting || !title.trim() ? '#ccc' : '#007bff',
                color: '#fff',
                cursor: isSubmitting || !title.trim() ? 'not-allowed' : 'pointer',
              }}
            >
              {isSubmitting ? 'Creating...' : 'Create Negotiation'}
            </button>
          </div>
        </form>
      </div>

      <style>{`
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .modal-content {
          background: white;
          border-radius: 8px;
          width: 90%;
          max-width: 500px;
          max-height: 90vh;
          overflow: auto;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px 24px;
          border-bottom: 1px solid #e0e0e0;
        }

        .modal-header h2 {
          margin: 0;
          font-size: 20px;
          font-weight: 600;
          color: #333;
        }

        .modal-close {
          background: none;
          border: none;
          font-size: 28px;
          color: #999;
          cursor: pointer;
          padding: 0;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 4px;
          transition: background 0.2s;
        }

        .modal-close:hover {
          background: #f0f0f0;
          color: #333;
        }

        .modal-body {
          padding: 24px;
        }

        .form-group label {
          display: block;
          font-size: 14px;
          font-weight: 500;
          color: #333;
        }

        .modal-footer {
          padding: 16px 24px;
          border-top: 1px solid #e0e0e0;
          display: flex;
          justify-content: flex-end;
        }
      `}</style>
    </div>
  );
}
