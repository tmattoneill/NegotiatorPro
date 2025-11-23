/**
 * Partner Persona Step - Step 2 of onboarding
 * Creates a partner persona to negotiate with
 */
import { useState } from 'react';
import type { PartnerPersonaCreate } from '../../types/personas';

interface PartnerPersonaStepProps {
  onComplete: (persona: PartnerPersonaCreate) => void;
  onBack: () => void;
  onSkip: () => void;
}

export default function PartnerPersonaStep({ onComplete, onBack, onSkip }: PartnerPersonaStepProps) {
  const [formData, setFormData] = useState<PartnerPersonaCreate>({
    name: '',
    role_title: '',
    company: '',
    is_shared: false,
  });

  const [errors, setErrors] = useState<{ name?: string }>({});

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    if (errors.name) {
      setErrors({});
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.name.trim()) {
      setErrors({ name: 'Name is required' });
      return;
    }

    onComplete(formData);
  };

  return (
    <div className="onboarding-step">
      <div className="step-header">
        <h2>Add a Partner</h2>
        <p className="step-description">
          Create a profile for someone you'll be negotiating with. You can always add more partners later.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="step-form">
        <div className="form-group">
          <label htmlFor="name">
            Partner Name <span className="required">*</span>
          </label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            placeholder="e.g., Jane Doe"
            autoFocus
            required
            maxLength={255}
          />
          {errors.name && <span className="error-text">{errors.name}</span>}
          <small className="field-hint">The person you're negotiating with</small>
        </div>

        <div className="form-group">
          <label htmlFor="role_title">
            Their Role/Title <span className="optional">(optional)</span>
          </label>
          <input
            type="text"
            id="role_title"
            name="role_title"
            value={formData.role_title}
            onChange={handleChange}
            placeholder="e.g., Procurement Manager, Hiring Manager"
            maxLength={255}
          />
          <small className="field-hint">Their professional role</small>
        </div>

        <div className="form-group">
          <label htmlFor="company">
            Their Company <span className="optional">(optional)</span>
          </label>
          <input
            type="text"
            id="company"
            name="company"
            value={formData.company}
            onChange={handleChange}
            placeholder="e.g., BigCo Inc"
            maxLength={255}
          />
          <small className="field-hint">The organization they represent</small>
        </div>

        <div className="step-actions">
          <button type="button" className="btn-secondary" onClick={onBack}>
            Back
          </button>
          <div style={{ flex: 1 }} />
          <button type="button" className="btn-text" onClick={onSkip}>
            Skip for now
          </button>
          <button type="submit" className="btn-primary" disabled={!formData.name.trim()}>
            Continue
          </button>
        </div>
      </form>

      <style>{`
        .onboarding-step {
          padding: 0;
        }

        .step-header {
          margin-bottom: 32px;
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
          background: #3498db;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          transition: background 0.2s;
        }

        .btn-primary:hover:not(:disabled) {
          background: #2980b9;
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

        .btn-text {
          padding: 10px 16px;
          font-size: 14px;
          font-weight: 500;
          background: none;
          color: #7f8c8d;
          border: none;
          cursor: pointer;
          transition: color 0.2s;
        }

        .btn-text:hover {
          color: #2c3e50;
        }
      `}</style>
    </div>
  );
}
