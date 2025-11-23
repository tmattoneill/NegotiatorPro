/**
 * Onboarding Wizard - Multi-step wizard for new user setup
 * Guides users through persona and negotiation creation
 */
import { useState } from 'react';
import { useAuthStore } from '../store/authStore';
import { usePersonaStore } from '../store/personaStore';
import { useNegotiationStore } from '../store/negotiationStore';
import type { UserPersonaCreate, PartnerPersonaCreate } from '../types/personas';
import UserPersonaStep from './onboarding/UserPersonaStep';
import PartnerPersonaStep from './onboarding/PartnerPersonaStep';
import NegotiationStep from './onboarding/NegotiationStep';

interface OnboardingWizardProps {
  onComplete: () => void;
  onSkip: () => void;
}

type Step = 'user-persona' | 'partner-persona' | 'negotiation' | 'complete';

export default function OnboardingWizard({ onComplete, onSkip }: OnboardingWizardProps) {
  const [currentStep, setCurrentStep] = useState<Step>('user-persona');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Temporary storage for wizard data
  const [userPersonaData, setUserPersonaData] = useState<UserPersonaCreate | null>(null);
  const [partnerPersonaData, setPartnerPersonaData] = useState<PartnerPersonaCreate | null>(null);
  const [createdPartnerPersonaId, setCreatedPartnerPersonaId] = useState<string | null>(null);

  const { user } = useAuthStore();
  const { createUserPersona, createPartnerPersona } = usePersonaStore();
  const { createNegotiation } = useNegotiationStore();

  const handleUserPersonaComplete = async (data: UserPersonaCreate) => {
    if (!user?.id) {
      setError('User not authenticated');
      return;
    }

    // DEBOUNCE FIX: Set submitting state immediately (synchronously)
    if (isSubmitting) return;
    setIsSubmitting(true);
    setError(null);

    try {
      await createUserPersona(user.id, data);
      setUserPersonaData(data);
      setCurrentStep('partner-persona');
    } catch (err) {
      console.error('Failed to create user persona:', err);
      setError('Failed to create your profile. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePartnerPersonaComplete = async (data: PartnerPersonaCreate) => {
    if (!user?.id) {
      setError('User not authenticated');
      return;
    }

    // DEBOUNCE FIX: Set submitting state immediately (synchronously)
    if (isSubmitting) return;
    setIsSubmitting(true);
    setError(null);

    try {
      const createdPersona = await createPartnerPersona(user.id, data);
      setPartnerPersonaData(data);
      setCreatedPartnerPersonaId(createdPersona.id);
      setCurrentStep('negotiation');
    } catch (err) {
      console.error('Failed to create partner persona:', err);
      setError('Failed to create partner profile. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePartnerPersonaSkip = () => {
    setPartnerPersonaData(null);
    setCurrentStep('negotiation');
  };

  const handleNegotiationComplete = async (title: string, description: string) => {
    if (!user?.id) {
      setError('User not authenticated');
      return;
    }

    // DEBOUNCE FIX: Set submitting state immediately (synchronously)
    if (isSubmitting) return;
    setIsSubmitting(true);
    setError(null);

    // Track created partner for potential rollback
    let createdDefaultPartnerId: string | null = null;

    try {
      // Determine which partner persona to use
      let partnerIds: string[];

      if (createdPartnerPersonaId) {
        // Use the partner persona created in the wizard
        partnerIds = [createdPartnerPersonaId];
      } else {
        // User skipped partner creation - create a default one
        const defaultPartner = await createPartnerPersona(user.id, {
          name: 'Partner',
          is_shared: false,
        });
        createdDefaultPartnerId = defaultPartner.id;
        partnerIds = [defaultPartner.id];
      }

      // CRITICAL FIX: Try to create negotiation, rollback on failure
      try {
        await createNegotiation(user.id, {
          title,
          description: description || undefined,
          partner_persona_ids: partnerIds,
        });

        // Mark onboarding as complete
        localStorage.setItem('onboardingCompleted', 'true');
        onComplete();
      } catch (negotiationError) {
        // ROLLBACK: Delete the default partner if we created it and negotiation failed
        if (createdDefaultPartnerId) {
          const { deletePartnerPersona } = usePersonaStore.getState();
          try {
            await deletePartnerPersona(user.id, createdDefaultPartnerId);
            console.log('Rolled back default partner creation after negotiation failure');
          } catch (rollbackError) {
            console.error('Failed to rollback partner persona:', rollbackError);
          }
        }
        throw negotiationError; // Re-throw to be caught by outer catch
      }
    } catch (err) {
      console.error('Failed to create negotiation:', err);
      setError('Failed to create negotiation. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      // Mark that user closed onboarding (so we don't show it again immediately)
      localStorage.setItem('onboardingDismissed', 'true');
      onSkip();
    }
  };

  return (
    <div className="wizard-overlay">
      <div className="wizard-container">
        {/* Header */}
        <div className="wizard-header">
          <div className="wizard-title">
            <h1>Welcome to NegotiatorPro</h1>
            <p>Let's get you set up in just a few steps</p>
          </div>
          <button
            className="wizard-close"
            onClick={handleClose}
            disabled={isSubmitting}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {/* Progress */}
        <div className="wizard-progress">
          <div className="progress-steps">
            <div className={`progress-step ${currentStep === 'user-persona' ? 'active' : ''} ${userPersonaData ? 'completed' : ''}`}>
              <div className="step-number">1</div>
              <div className="step-label">Your Profile</div>
            </div>
            <div className="progress-line" />
            <div className={`progress-step ${currentStep === 'partner-persona' ? 'active' : ''} ${partnerPersonaData ? 'completed' : ''}`}>
              <div className="step-number">2</div>
              <div className="step-label">Add Partner</div>
            </div>
            <div className="progress-line" />
            <div className={`progress-step ${currentStep === 'negotiation' ? 'active' : ''}`}>
              <div className="step-number">3</div>
              <div className="step-label">First Negotiation</div>
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="wizard-error">
            {error}
          </div>
        )}

        {/* Content */}
        <div className="wizard-content">
          {currentStep === 'user-persona' && (
            <UserPersonaStep onComplete={handleUserPersonaComplete} />
          )}

          {currentStep === 'partner-persona' && (
            <PartnerPersonaStep
              onComplete={handlePartnerPersonaComplete}
              onBack={() => setCurrentStep('user-persona')}
              onSkip={handlePartnerPersonaSkip}
            />
          )}

          {currentStep === 'negotiation' && (
            <NegotiationStep
              userPersonaName={userPersonaData?.name || 'You'}
              partnerPersonaName={partnerPersonaData?.name}
              onComplete={handleNegotiationComplete}
              onBack={() => setCurrentStep('partner-persona')}
            />
          )}
        </div>

        {/* Loading Overlay */}
        {isSubmitting && (
          <div className="wizard-loading">
            <div className="loading-spinner" />
            <p>Setting up your account...</p>
          </div>
        )}
      </div>

      <style>{`
        .wizard-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 9999;
          padding: 20px;
        }

        .wizard-container {
          background: white;
          border-radius: 12px;
          width: 100%;
          max-width: 600px;
          max-height: 90vh;
          overflow: auto;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
          position: relative;
        }

        .wizard-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          padding: 32px 32px 24px 32px;
          border-bottom: 1px solid #ecf0f1;
        }

        .wizard-title h1 {
          margin: 0 0 8px 0;
          font-size: 28px;
          font-weight: 700;
          color: #2c3e50;
        }

        .wizard-title p {
          margin: 0;
          font-size: 15px;
          color: #7f8c8d;
        }

        .wizard-close {
          background: none;
          border: none;
          font-size: 32px;
          color: #95a5a6;
          cursor: pointer;
          padding: 0;
          width: 36px;
          height: 36px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 6px;
          transition: all 0.2s;
          flex-shrink: 0;
          margin-left: 16px;
        }

        .wizard-close:hover:not(:disabled) {
          background: #f8f9fa;
          color: #2c3e50;
        }

        .wizard-close:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .wizard-progress {
          padding: 24px 32px;
          background: #f8f9fa;
        }

        .progress-steps {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .progress-step {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
          flex: 0 0 auto;
        }

        .step-number {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: white;
          border: 2px solid #dfe6e9;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 16px;
          font-weight: 600;
          color: #95a5a6;
          transition: all 0.3s;
        }

        .progress-step.active .step-number {
          background: #3498db;
          border-color: #3498db;
          color: white;
        }

        .progress-step.completed .step-number {
          background: #27ae60;
          border-color: #27ae60;
          color: white;
        }

        .step-label {
          font-size: 13px;
          color: #95a5a6;
          font-weight: 500;
          white-space: nowrap;
        }

        .progress-step.active .step-label {
          color: #2c3e50;
        }

        .progress-line {
          flex: 1;
          height: 2px;
          background: #dfe6e9;
          margin: 0 12px;
          margin-bottom: 30px;
        }

        .wizard-error {
          margin: 16px 32px;
          padding: 12px 16px;
          background: #fee;
          border: 1px solid #fcc;
          border-radius: 6px;
          color: #c33;
          font-size: 14px;
        }

        .wizard-content {
          padding: 32px;
          min-height: 300px;
        }

        .wizard-loading {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(255, 255, 255, 0.95);
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 16px;
          border-radius: 12px;
        }

        .loading-spinner {
          width: 40px;
          height: 40px;
          border: 3px solid #ecf0f1;
          border-top-color: #3498db;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        .wizard-loading p {
          margin: 0;
          font-size: 15px;
          color: #7f8c8d;
        }

        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }

        @media (max-width: 640px) {
          .wizard-container {
            max-width: 100%;
            border-radius: 0;
            max-height: 100vh;
          }

          .wizard-header {
            padding: 24px 20px 20px 20px;
          }

          .wizard-title h1 {
            font-size: 24px;
          }

          .wizard-progress {
            padding: 20px;
          }

          .wizard-content {
            padding: 24px 20px;
          }

          .step-label {
            font-size: 11px;
          }
        }
      `}</style>
    </div>
  );
}
