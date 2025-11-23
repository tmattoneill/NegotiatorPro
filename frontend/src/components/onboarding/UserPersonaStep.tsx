/**
 * User Persona Step - Step 1 of onboarding
 * Creates the user's own persona
 */
import { useState } from 'react';
import type { UserPersonaCreate } from '../../types/personas';

interface UserPersonaStepProps {
  onComplete: (persona: UserPersonaCreate) => void;
  onSkip?: () => void;
}

export default function UserPersonaStep({ onComplete }: UserPersonaStepProps) {
  const [formData, setFormData] = useState<UserPersonaCreate>({
    name: '',
    role_title: '',
    organization: '',
    is_default: true,
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
        <h2>Create Your Profile</h2>
        <p className="step-description">
          Let's start by setting up your negotiator profile. This helps personalize your experience.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="step-form">
        <div className="form-group">
          <label htmlFor="name">
            Your Name <span className="required">*</span>
          </label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            placeholder="e.g., John Smith"
            autoFocus
            required
            maxLength={255}
          />
          {errors.name && <span className="error-text">{errors.name}</span>}
          <small className="field-hint">How you'd like to be addressed</small>
        </div>

        <div className="form-group">
          <label htmlFor="role_title">
            Role/Title <span className="optional">(optional)</span>
          </label>
          <input
            type="text"
            id="role_title"
            name="role_title"
            value={formData.role_title}
            onChange={handleChange}
            placeholder="e.g., Sales Director, Product Manager"
            maxLength={255}
          />
          <small className="field-hint">Your professional role</small>
        </div>

        <div className="form-group">
          <label htmlFor="organization">
            Organization <span className="optional">(optional)</span>
          </label>
          <input
            type="text"
            id="organization"
            name="organization"
            value={formData.organization}
            onChange={handleChange}
            placeholder="e.g., Acme Corp"
            maxLength={255}
          />
          <small className="field-hint">Your company or organization</small>
        </div>

        <div className="step-actions">
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
      `}</style>
    </div>
  );
}
