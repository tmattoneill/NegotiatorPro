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
    <div className="space-y-6">
      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-chat-foreground mb-2">Add a Partner</h2>
        <p className="text-[15px] text-chat-muted-foreground leading-relaxed">
          Create a profile for someone you'll be negotiating with. You can always add more partners later.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <label htmlFor="name" className="text-[14px] font-medium text-chat-foreground">
            Partner Name <span className="text-danger">*</span>
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
            className="px-3 py-2 text-[14px] border border-chat-border rounded focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 outline-none"
          />
          {errors.name && <span className="text-[13px] text-danger">{errors.name}</span>}
          <small className="text-[13px] text-chat-muted-foreground -mt-1">The person you're negotiating with</small>
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="role_title" className="text-[14px] font-medium text-chat-foreground">
            Their Role/Title <span className="text-chat-muted-foreground font-normal text-[13px]">(optional)</span>
          </label>
          <input
            type="text"
            id="role_title"
            name="role_title"
            value={formData.role_title}
            onChange={handleChange}
            placeholder="e.g., Procurement Manager, Hiring Manager"
            maxLength={255}
            className="px-3 py-2 text-[14px] border border-chat-border rounded focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 outline-none"
          />
          <small className="text-[13px] text-chat-muted-foreground -mt-1">Their professional role</small>
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="company" className="text-[14px] font-medium text-chat-foreground">
            Their Company <span className="text-chat-muted-foreground font-normal text-[13px]">(optional)</span>
          </label>
          <input
            type="text"
            id="company"
            name="company"
            value={formData.company}
            onChange={handleChange}
            placeholder="e.g., BigCo Inc"
            maxLength={255}
            className="px-3 py-2 text-[14px] border border-chat-border rounded focus:border-chat-primary focus:ring-2 focus:ring-chat-primary/10 outline-none"
          />
          <small className="text-[13px] text-chat-muted-foreground -mt-1">The organization they represent</small>
        </div>

        <div className="flex items-center justify-end gap-3 mt-2 pt-6 border-t border-chat-border">
          <button type="button" className="px-4 py-2 rounded border border-chat-border text-chat-muted-foreground bg-chat-card hover:bg-chat-muted" onClick={onBack}>
            Back
          </button>
          <div className="flex-1" />
          <button type="button" className="px-3 py-2 rounded bg-transparent text-chat-muted-foreground hover:text-chat-foreground" onClick={onSkip}>
            Skip for now
          </button>
          <button type="submit" className="px-5 py-2.5 rounded-md bg-chat-primary text-white disabled:opacity-50" disabled={!formData.name.trim()}>
            Continue
          </button>
        </div>
      </form>
    </div>
  );
}
