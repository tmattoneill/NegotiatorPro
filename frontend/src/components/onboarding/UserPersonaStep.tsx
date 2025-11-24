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
    <div className="space-y-6">
      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-chat-foreground mb-2">Create Your Profile</h2>
        <p className="text-[15px] text-chat-muted-foreground leading-relaxed">
          Let's start by setting up your negotiator profile. This helps personalize your experience.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <label htmlFor="name" className="text-[14px] font-medium text-chat-foreground">
            Your Name <span className="text-danger">*</span>
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
            className="px-3 py-2 text-[14px] border border-chat-border rounded focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 outline-none"
          />
          {errors.name && <span className="text-[13px] text-danger">{errors.name}</span>}
          <small className="text-[13px] text-chat-muted-foreground -mt-1">How you'd like to be addressed</small>
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="role_title" className="text-[14px] font-medium text-chat-foreground">
            Role/Title <span className="text-chat-muted-foreground font-normal text-[13px]">(optional)</span>
          </label>
          <input
            type="text"
            id="role_title"
            name="role_title"
            value={formData.role_title}
            onChange={handleChange}
            placeholder="e.g., Sales Director, Product Manager"
            maxLength={255}
            className="px-3 py-2 text-[14px] border border-chat-border rounded focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 outline-none"
          />
          <small className="text-[13px] text-chat-muted-foreground -mt-1">Your professional role</small>
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="organization" className="text-[14px] font-medium text-chat-foreground">
            Organization <span className="text-chat-muted-foreground font-normal text-[13px]">(optional)</span>
          </label>
          <input
            type="text"
            id="organization"
            name="organization"
            value={formData.organization}
            onChange={handleChange}
            placeholder="e.g., Acme Corp"
            maxLength={255}
            className="px-3 py-2 text-[14px] border border-chat-border rounded focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 outline-none"
          />
          <small className="text-[13px] text-chat-muted-foreground -mt-1">Your company or organization</small>
        </div>

        <div className="flex justify-end gap-3 mt-2 pt-6 border-t border-chat-border">
          <button type="submit" className="px-5 py-2.5 rounded-md bg-chat-primary text-white disabled:opacity-50" disabled={!formData.name.trim()}>
            Continue
          </button>
        </div>
      </form>
    </div>
  );
}
