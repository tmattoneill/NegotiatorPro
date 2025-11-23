# New User Onboarding Flow Implementation Plan

## Overview
Create a seamless onboarding experience for new users that guides them through persona and negotiation setup, eliminating the "dead-end" where new users can't create negotiations due to missing personas.

## Current State Analysis

### User Registration Flow
**Endpoint**: `POST /api/users/`
**File**: `backend/api/routes/users.py`

**Required Fields**:
- `username` (min 3 chars, max 255, unique)
- `email` (valid email format, unique)
- `password` (min 8 chars)

**Optional Fields**:
- `first_name`, `last_name`
- `openai_api_key`, `anthropic_api_key` (encrypted)
- `role` (default: "user")

### The Problem

When a new user registers, they have:
- ✅ User account created
- ❌ NO user personas
- ❌ NO partner personas
- ❌ NO negotiations
- ❌ NO conversations

**Current User Journey** (Broken):
1. User registers → lands on main app
2. Sees empty state: "No negotiations yet - click + New"
3. Clicks "+ New" to create negotiation
4. **ERROR**: "No partner personas available. Please create a partner persona first from your profile."
5. Must navigate to `/profile` → Find persona manager → Create user persona → Create partner persona
6. Return to create negotiation
7. Only then can start conversations

**Root Cause**: App assumes personas already exist. New users hit a dead-end.

### Database Dependencies

```
User Registration
  ↓
User Profile Created (isolated)
  ↓
Manual: Create User Persona (represent yourself)
  ↓
Manual: Create Partner Persona (represent counterpart)
  ↓
Create Negotiation (requires at least 1 partner_persona_id)
  ↓
Create Conversation (chat session within negotiation)
  ↓
Send/Receive Messages
```

### Possible 400 Error Causes

From the console logs showing `POST /api/users/ 400 (Bad Request)`:

1. **Duplicate username/email**: Most likely cause
   ```python
   # UserProfileManager.create_user()
   if existing:
       raise ValueError("Username or email already exists")
   ```

2. **Validation failures**:
   - Username < 3 chars or > 255 chars
   - Invalid email format
   - Password < 8 chars
   - Invalid role value

## Implementation Plan

### Phase 1: Fix User Registration API Issue (400 Error)
**Goal**: Ensure user creation works properly

1. **Investigate and Fix 400 Error**
   - Check backend logs for specific validation error
   - Add better error messages to frontend form
   - Show which field is causing the issue (username/email duplicate)
   - Files:
     - `backend/api/routes/users.py`
     - `frontend/src/components/CreateProfile.tsx`

### Phase 2: Create Onboarding Wizard Component
**Goal**: Guide new users through required setup steps

2. **Create OnboardingWizard.tsx Component**
   - Multi-step wizard modal
   - Step 1: Create User Persona (who you are)
   - Step 2: Create Partner Persona (who you're negotiating with)
   - Step 3: Create First Negotiation
   - Shows on first login (check if user has 0 negotiations)
   - File: `frontend/src/components/OnboardingWizard.tsx` (new)

3. **Step 1: User Persona Form**
   - **Required**: Name
   - **Optional**: Role title, organization, communication style, strengths
   - Auto-set as default persona (`is_default: true`)
   - Simple, friendly UI with helpful placeholders
   - Example: "John Smith, Sales Director at Acme Corp"

4. **Step 2: Partner Persona Form**
   - **Required**: Name
   - **Optional**: Company, role, communication style, interests
   - Option to mark as "shared" (visible to all users)
   - Skip button if they want to create later
   - Example: "Jane Doe, Procurement Manager at BigCo"

5. **Step 3: First Negotiation Form**
   - **Required**: Negotiation title
   - **Optional**: Description
   - Auto-assigns user persona from Step 1
   - Auto-assigns partner persona from Step 2
   - Creates negotiation and first conversation automatically
   - Example: "Salary Negotiation - Q4 2024"

### Phase 3: Integrate Onboarding into App Flow

6. **Update App.tsx**
   - Add onboarding detection logic
   - Show wizard if: `user.id exists AND negotiations.length === 0`
   - Track completion in localStorage: `onboardingCompleted: true`
   - Don't show wizard if user manually closed it before

7. **Update CreateProfile.tsx**
   - After successful registration, set flag: `needsOnboarding: true`
   - Redirect to main app (wizard will trigger automatically)
   - Better error handling for duplicate username/email

8. **Update NegotiationModal.tsx**
   - Remove hard error for missing partner personas
   - Instead: Show inline "Create Partner" mini-form if none exist
   - Or provide quick link to persona creation
   - Make it seamless, not a dead-end

### Phase 4: Backend Support (Optional Improvements)

9. **Optional: Auto-create Default Personas on Registration**
   - When user registers, automatically create:
     - Default user persona: "My Profile" (empty details, `is_default: true`)
     - Shared default partner: "General Partner" (`is_shared: true`)
   - Allows immediate negotiation creation without wizard
   - File: `backend/api/routes/users.py` (modify create_user endpoint)
   - Trade-off: Cleaner DB vs. simpler UX

10. **Optional: Persona Quick-Create Endpoint**
    - New endpoint: `POST /api/personas/quick-setup`
    - Takes minimal data, creates both user + partner personas in one call
    - Returns both IDs for immediate negotiation creation
    - Reduces API calls during onboarding

### Phase 5: UX Polish

11. **Empty State Improvements**
    - When `negotiations.length === 0`, show welcoming empty state
    - Large call-to-action: "Welcome! Start Your First Negotiation"
    - Guide users to wizard or inline persona creation
    - File: `frontend/src/components/Sidebar.tsx`

12. **Error Handling**
    - Clear error messages at each onboarding step
    - Allow users to skip/come back later
    - Save partial progress (e.g., user persona created but not partner)
    - "You can always finish this from your Profile later"

13. **Progress Indicators**
    - Show wizard step progress: "Step 1 of 3"
    - Visual checkmarks for completed steps
    - Option to go back and edit previous steps

## Files to Create

### New Files:
- `frontend/src/components/OnboardingWizard.tsx` - Main wizard component
- `frontend/src/components/onboarding/UserPersonaStep.tsx` - Step 1 form
- `frontend/src/components/onboarding/PartnerPersonaStep.tsx` - Step 2 form
- `frontend/src/components/onboarding/NegotiationStep.tsx` - Step 3 form
- `dev-docs/NEW-USER.md` - This document (implementation plan)

## Files to Modify

### Frontend:
- `frontend/src/App.tsx` - Add onboarding trigger logic
- `frontend/src/components/CreateProfile.tsx` - Better error handling, set onboarding flag
- `frontend/src/components/NegotiationModal.tsx` - Inline persona creation option
- `frontend/src/components/Sidebar.tsx` - Better empty state messaging
- `frontend/src/store/negotiationStore.ts` - ✅ Already fixed (stale ID validation)

### Backend:
- `backend/api/routes/users.py` - Fix 400 error + optional auto-persona creation
- `backend/api/routes/personas.py` - Optional quick-setup endpoint

## Data Models Reference

### User Persona (Step 1)
```json
{
  "name": "John Smith",
  "role_title": "Sales Director",
  "organization": "Acme Corp",
  "communication_style": "Direct and assertive",
  "negotiation_strengths": "Strong analytical skills",
  "notes": "Prefers data-driven arguments",
  "is_default": true
}
```

### Partner Persona (Step 2)
```json
{
  "name": "Jane Doe",
  "role_title": "Procurement Manager",
  "company": "BigCo Inc",
  "communication_style": "Collaborative",
  "known_interests": "Cost reduction, long-term relationships",
  "batna_estimate": "Alternative suppliers available",
  "relationship_notes": "Met at industry conference",
  "is_shared": false
}
```

### Negotiation (Step 3)
```json
{
  "title": "Salary Negotiation - Q4 2024",
  "description": "Annual review discussion",
  "user_persona_id": "uuid-from-step-1",
  "partner_persona_ids": ["uuid-from-step-2"],
  "primary_partner_id": "uuid-from-step-2",
  "settings": {}
}
```

## Success Criteria

✅ New users can register without 400 errors
✅ After registration, guided wizard appears automatically
✅ User creates persona, partner persona, and negotiation in ~3 steps
✅ First conversation auto-created on negotiation creation
✅ User lands in chat interface ready to send first message
✅ Existing users unaffected by changes
✅ Users can skip wizard and come back later if needed
✅ Clear error messages at each step
✅ Progress saved if user abandons wizard

## Testing Checklist

### Registration Flow:
- [ ] Create new user account (no 400 errors)
- [ ] Duplicate username shows clear error
- [ ] Duplicate email shows clear error
- [ ] Invalid email format rejected
- [ ] Password validation works

### Onboarding Wizard:
- [ ] Wizard appears automatically after first login
- [ ] Wizard doesn't appear for existing users
- [ ] Can complete Step 1 (user persona) with minimal info
- [ ] Can complete Step 2 (partner persona) with minimal info
- [ ] Can skip Step 2 and create partner later
- [ ] Can complete Step 3 (negotiation) successfully
- [ ] First conversation appears in sidebar
- [ ] Can close wizard and resume later

### Chat Functionality:
- [ ] Can send first message immediately after onboarding
- [ ] Can receive response from model
- [ ] Messages persist after refresh
- [ ] Can create additional conversations
- [ ] Can create additional negotiations

### Edge Cases:
- [ ] User closes wizard before completion
- [ ] User navigates away during wizard
- [ ] API errors handled gracefully
- [ ] Network errors handled gracefully
- [ ] Multiple tabs/windows open

## Related Files Reference

### Backend:
- User creation: `backend/api/routes/users.py`
- User manager: `backend/user_profile.py`
- Persona routes: `backend/api/routes/personas.py`
- Persona CRUD: `backend/db_operations.py`
- Negotiation routes: `backend/api/routes/negotiations.py`
- Auth routes: `backend/api/routes/auth.py`

### Frontend:
- Registration form: `frontend/src/components/CreateProfile.tsx`
- Login form: `frontend/src/components/Login.tsx`
- Main app: `frontend/src/App.tsx`
- Sidebar: `frontend/src/components/Sidebar.tsx`
- Negotiation modal: `frontend/src/components/NegotiationModal.tsx`
- Chat container: `frontend/src/components/ChatContainer.tsx`

### Database:
- Full schema: `migrations/001_full_schema.sql`

### Models:
- Persona models: `backend/api/models/personas.py`
- Negotiation models: `backend/api/models/negotiations.py`

## Timeline Estimate

- **Phase 1** (Fix 400 error): 30 minutes
- **Phase 2** (Wizard component): 2-3 hours
- **Phase 3** (Integration): 1-2 hours
- **Phase 4** (Backend improvements): 1-2 hours (optional)
- **Phase 5** (Polish): 1 hour

**Total**: 5-8 hours for complete implementation

## Notes

- Prioritize Phase 1-3 for MVP
- Phase 4-5 can be added incrementally
- Consider A/B testing wizard vs. auto-persona creation
- Monitor onboarding completion rates
- Collect user feedback on wizard UX
