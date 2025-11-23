/**
 * Negotiation Step - Step 3 of onboarding
 * Creates the first negotiation
 */
import { useState } from 'react';

interface NegotiationStepProps {
  userPersonaName: string;
  partnerPersonaName?: string;
  onComplete: (title: string, description: string) => void;
  onBack: () => void;
}

export default function NegotiationStep({
  userPersonaName,
  partnerPersonaName,
  onComplete,
  onBack,
}: NegotiationStepProps) {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [errors, setErrors] = useState<{ title?: string }>({});

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!title.trim()) {
      setErrors({ title: 'Negotiation name is required' });
      return;
    }

    onComplete(title.trim(), description.trim());
  };

  return (
    <div className="onboarding-step">
      <div className="step-header">
        <h2>Create Your First Negotiation</h2>
        <p className="step-description">
          Give your negotiation a name. This could be a job offer, contract, or any situation where you need guidance.
        </p>
      </div>

      <div className="persona-summary">
        <div className="persona-item">
          <span className="persona-label">You:</span>
          <span className="persona-name">{userPersonaName}</span>
        </div>
        {partnerPersonaName && (
          <div className="persona-item">
            <span className="persona-label">Partner:</span>
            <span className="persona-name">{partnerPersonaName}</span>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="step-form">
        <div className="form-group">
          <label htmlFor="title">
            Negotiation Name <span className="required">*</span>
          </label>
          <input
            type="text"
            id="title"
            name="title"
            value={title}
            onChange={(e) => {
              setTitle(e.target.value);
              if (errors.title) setErrors({});
            }}
            placeholder="e.g., Salary Negotiation, Vendor Contract"
            autoFocus
            required
            maxLength={255}
          />
          {errors.title && <span className="error-text">{errors.title}</span>}
          <small className="field-hint">A clear, descriptive name for this negotiation</small>
        </div>

        <div className="form-group">
          <label htmlFor="description">
            Description <span className="optional">(optional)</span>
          </label>
          <textarea
            id="description"
            name="description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Brief context about the negotiation..."
            rows={4}
          />
          <small className="field-hint">Any additional context or goals</small>
        </div>

        <div className="step-actions">
          <button type="button" className="btn-secondary" onClick={onBack}>
            Back
          </button>
          <div style={{ flex: 1 }} />
          <button type="submit" className="btn-primary" disabled={!title.trim()}>
            Start Negotiating
          </button>
        </div>
      </form>

      <style>{`
        .onboarding-step {
          padding: 0;
        }

        .step-header {
          margin-bottom: 24px;
        }

        .step-header h2 {
          margin: 0 0 12px 0;
          font-size: 24px;
          font-weight: 600;
          color: #2c3e50;
        }

        .step-description {
          margin: 0;
          font-size: 15px;
          color: #7f8c8d;
          line-height: 1.5;
        }

        .persona-summary {
          background: #f8f9fa;
          border: 1px solid #e9ecef;
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 32px;
        }

        .persona-item {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }

        .persona-item:last-child {
          margin-bottom: 0;
        }

        .persona-label {
          font-size: 14px;
          font-weight: 500;
          color: #7f8c8d;
          min-width: 70px;
        }

        .persona-name {
          font-size: 14px;
          color: #2c3e50;
          font-weight: 500;
        }

        .step-form {
          display: flex;
          flex-direction: column;
          gap: 24px;
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .form-group label {
          font-size: 14px;
          font-weight: 500;
          color: #2c3e50;
        }

        .form-group input,
        .form-group textarea {
          padding: 10px 12px;
          font-size: 14px;
          border: 1px solid #dfe6e9;
          border-radius: 6px;
          transition: border-color 0.2s;
          font-family: inherit;
        }

        .form-group input:focus,
        .form-group textarea:focus {
          outline: none;
          border-color: #3498db;
        }

        .form-group textarea {
          resize: vertical;
          min-height: 100px;
        }

        .required {
          color: #e74c3c;
        }

        .optional {
          color: #95a5a6;
          font-weight: normal;
          font-size: 13px;
        }

        .field-hint {
          font-size: 13px;
          color: #95a5a6;
          margin-top: -4px;
        }

        .error-text {
          font-size: 13px;
          color: #e74c3c;
        }

        .step-actions {
          display: flex;
          justify-content: flex-end;
          align-items: center;
          gap: 12px;
          margin-top: 8px;
          padding-top: 24px;
          border-top: 1px solid #ecf0f1;
        }

        .btn-primary {
          padding: 10px 24px;
          font-size: 14px;
          font-weight: 500;
          background: #27ae60;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          transition: background 0.2s;
        }

        .btn-primary:hover:not(:disabled) {
          background: #229954;
        }

        .btn-primary:disabled {
          background: #bdc3c7;
          cursor: not-allowed;
        }

        .btn-secondary {
          padding: 10px 24px;
          font-size: 14px;
          font-weight: 500;
          background: white;
          color: #7f8c8d;
          border: 1px solid #dfe6e9;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn-secondary:hover {
          border-color: #bdc3c7;
          background: #f8f9fa;
        }
      `}</style>
    </div>
  );
}
