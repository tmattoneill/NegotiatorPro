/**
 * Main App component - clean layout with sidebar
 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from './store/authStore';
import { useAdminStore } from './store/adminStore';
import { useNegotiationStore } from './store/negotiationStore';
import { usePersonaStore } from './store/personaStore';
import { useChatStore } from './store/chatStore';
import Sidebar from './components/Sidebar';
import ChatContainer from './components/ChatContainer';
import SystemPromptEditor from './components/SystemPromptEditor';
import AdminPanel from './components/AdminPanel';
import OnboardingWizard from './components/OnboardingWizard';

function App() {
  const navigate = useNavigate();
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const user = useAuthStore((state) => state.user);
  const adminView = useAdminStore((state) => state.currentView);
  const loadNegotiations = useNegotiationStore((state) => state.loadNegotiations);
  const fetchUserPersonas = usePersonaStore((state) => state.fetchUserPersonas);
  const fetchPartnerPersonas = usePersonaStore((state) => state.fetchPartnerPersonas);

  const [showOnboarding, setShowOnboarding] = useState(false);
  const [isCheckingOnboarding, setIsCheckingOnboarding] = useState(true);

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!isAuthenticated) {
      navigate('/login');

      // CRITICAL FIX: Clear ALL stores and flags on logout
      // This prevents data leakage between user sessions

      // IMPORTANT: Clear localStorage FIRST (before state mutations)
      // This ensures persist middleware doesn't restore stale data
      localStorage.removeItem('onboardingCompleted');
      localStorage.removeItem('onboardingDismissed');
      localStorage.removeItem('negotiation-storage');
      localStorage.removeItem('chat-storage');

      // Then clear in-memory state
      // Using store methods instead of direct mutation for better reactivity
      const negotiationStore = useNegotiationStore.getState();
      if (negotiationStore.setCurrentNegotiation) {
        negotiationStore.setCurrentNegotiation(null);
      }
      negotiationStore.negotiations = [];
      negotiationStore.currentNegotiationId = null;

      const chatState = useChatStore.getState();
      chatState.sessions = [];
      chatState.currentSessionId = null;

      const personaState = usePersonaStore.getState();
      personaState.userPersonas = [];
      personaState.partnerPersonas = [];
      personaState.selectedUserPersona = null;
      personaState.selectedPartnerPersona = null;
    }
  }, [isAuthenticated, navigate]);

  useEffect(() => {
    // Check if user needs onboarding
    const checkOnboarding = async () => {
      if (!user?.id || !isAuthenticated) {
        setIsCheckingOnboarding(false);
        return;
      }

      // DEFENSIVE FIX: Clear any stale localStorage on fresh login
      // This handles cases where user refreshed page before logout cleanup ran
      const storedData = localStorage.getItem('negotiation-storage');
      if (storedData) {
        try {
          JSON.parse(storedData); // Validate it's valid JSON
          // If localStorage has data but user just logged in, it's stale
          // We'll let the API calls below overwrite it with fresh data
        } catch (e) {
          // Invalid data, clear it
          localStorage.removeItem('negotiation-storage');
        }
      }

      // Check if onboarding was already completed or dismissed
      const onboardingCompleted = localStorage.getItem('onboardingCompleted');
      const onboardingDismissed = localStorage.getItem('onboardingDismissed');

      if (onboardingCompleted || onboardingDismissed) {
        setIsCheckingOnboarding(false);
        return;
      }

      // Load negotiations and personas to check if user needs onboarding
      try {
        await Promise.all([
          loadNegotiations(user.id),
          fetchUserPersonas(user.id),
          fetchPartnerPersonas(user.id),
        ]);

        // CRITICAL FIX: Read fresh state from store after async load completes
        const freshNegotiations = useNegotiationStore.getState().negotiations;

        // Show onboarding if user has no negotiations
        if (freshNegotiations.length === 0) {
          setShowOnboarding(true);
        }
      } catch (error) {
        console.error('Failed to check onboarding status:', error);
      } finally {
        setIsCheckingOnboarding(false);
      }
    };

    checkOnboarding();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id, isAuthenticated]);

  const handleOnboardingComplete = async () => {
    setShowOnboarding(false);

    // CRITICAL FIX: Reload ALL data to ensure fresh state after onboarding
    if (user?.id) {
      await Promise.all([
        loadNegotiations(user.id),
        fetchUserPersonas(user.id),
        fetchPartnerPersonas(user.id),
      ]);

      // After loading, get the fresh negotiations and set the current one
      const freshNegotiations = useNegotiationStore.getState().negotiations;
      if (freshNegotiations.length > 0) {
        const firstNegotiation = freshNegotiations[0];

        // Auto-select the first (newly created) negotiation
        useNegotiationStore.getState().setCurrentNegotiation(firstNegotiation.id);

        // CRITICAL FIX: Explicitly load conversations for the selected negotiation
        // This ensures the sidebar shows conversations immediately
        const chatStore = useChatStore.getState();
        if (chatStore.loadConversations) {
          await chatStore.loadConversations(firstNegotiation.id, user.id);
        }
      }
    }
  };

  const handleOnboardingSkip = () => {
    setShowOnboarding(false);
  };

  // Don't render the app if not authenticated (will redirect)
  if (!isAuthenticated) {
    return null;
  }

  // Show loading while checking onboarding status
  if (isCheckingOnboarding) {
    return (
      <div className="app-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', color: '#7f8c8d' }}>
          <div style={{
            width: '40px',
            height: '40px',
            border: '3px solid #ecf0f1',
            borderTopColor: '#3498db',
            borderRadius: '50%',
            animation: 'spin 0.8s linear infinite',
            margin: '0 auto 16px'
          }} />
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  // Render appropriate view based on admin state
  const renderMainContent = () => {
    switch (adminView) {
      case 'system-prompt':
        return <SystemPromptEditor />;
      case 'admin-panel':
        return <AdminPanel />;
      default:
        return <ChatContainer />;
    }
  };

  return (
    <>
      <div className="app-container">
        <Sidebar />
        {renderMainContent()}
      </div>

      {/* Onboarding Wizard */}
      {showOnboarding && (
        <OnboardingWizard
          onComplete={handleOnboardingComplete}
          onSkip={handleOnboardingSkip}
        />
      )}
    </>
  );
}

export default App;
