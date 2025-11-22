# Chat Not Working - Root Cause & Fixes

## Problem Statement
User reported: "Chat input does nothing when I type and press Enter. No errors in console, no network activity."

## Root Cause Analysis

### Application Architecture
The app requires this hierarchy to function:
```
User → Negotiation → Conversation → Messages
```

### Issues Identified

1. **Missing Negotiation Selector UI**
   - Negotiations were being loaded in background (`Sidebar.tsx` line 33)
   - No UI component to display or select negotiations
   - Users had no way to activate a negotiation

2. **Silent Failures Without User Feedback**
   - `ChatContainer.tsx:31-33` silently returned if no `currentSession?.id`
   - "New Conversation" button was disabled without tooltip explaining why
   - No indication that negotiation selection was required

3. **Field Name Mismatch**
   - API returns `title` field for negotiations
   - Frontend code was checking `neg.name` (legacy field)
   - Caused "Untitled Negotiation" to display instead of actual name

4. **Missing userId Parameter in API Call**
   - Backend endpoint `/api/conversations/negotiation/{id}` requires `user_id` query param
   - Frontend `getNegotiationConversations()` wasn't passing userId
   - API returned validation error: "Required field 'user_id' is missing"

## Fixes Applied

### 1. Added Negotiation Selector to Sidebar
**File**: `frontend/src/components/Sidebar.tsx`

**Changes**:
```typescript
// Added to imports (line 20)
const { currentNegotiation, negotiations, loadNegotiations, setCurrentNegotiation } = useNegotiationStore();

// Added new UI section after personas display (around line 104)
<div style={{ padding: '12px 16px', background: 'rgba(255, 255, 255, 0.05)', ... }}>
  <div style={{ fontSize: '10px', ... }}>Active Negotiation</div>
  {negotiations.length > 0 ? (
    <select
      value={currentNegotiation?.id || ''}
      onChange={(e) => {
        console.log('Negotiation selected:', e.target.value);
        setCurrentNegotiation(e.target.value);
      }}
    >
      <option value="">Select a negotiation...</option>
      {negotiations.map((neg) => (
        <option key={neg.id} value={neg.id}>
          {neg.title || neg.name || 'Untitled Negotiation'}
        </option>
      ))}
    </select>
  ) : (
    <div>No negotiations yet</div>
  )}
</div>
```

**Added tooltip to "New Conversation" button** (line 176):
```typescript
title={!currentNegotiation?.id ? 'Please select a negotiation first' : 'Create a new conversation'}
```

**Added debug logging to useEffect** (line 38-44):
```typescript
useEffect(() => {
  console.log('Negotiation changed:', currentNegotiation?.id);
  if (currentNegotiation?.id && user?.id) {
    console.log('Loading conversations for negotiation:', currentNegotiation.id);
    loadConversations(currentNegotiation.id, user.id);
  }
}, [currentNegotiation?.id, user?.id]);
```

### 2. Improved Error Feedback in Chat
**File**: `frontend/src/components/ChatContainer.tsx`

**Changes** (line 31-42):
```typescript
const handleSendMessage = async (content: string, files?: File[]) => {
  if (!currentSession?.id) {
    console.warn('No active conversation - message not sent');
    const errorMessage: Message = {
      id: Date.now().toString(),
      role: 'assistant',
      content: '⚠️ Please create or select a conversation first. Click "New Conversation" in the sidebar.',
      timestamp: new Date(),
    };
    addMessage(errorMessage);
    return;
  }
  // ... rest of function
}
```

### 3. Fixed API Service to Pass userId
**File**: `frontend/src/services/api.ts`

**Changes** (line 177-180):
```typescript
// Before:
export const getNegotiationConversations = async (negotiationId: string): Promise<Conversation[]> => {
  const response = await api.get<Conversation[]>(`/conversations/negotiation/${negotiationId}`);
  return response.data;
};

// After:
export const getNegotiationConversations = async (negotiationId: string, userId: string): Promise<Conversation[]> => {
  const response = await api.get<Conversation[]>(`/conversations/negotiation/${negotiationId}?user_id=${userId}`);
  return response.data;
};
```

### 4. Updated Chat Store to Pass userId
**File**: `frontend/src/store/chatStore.ts`

**Changes** (line 40):
```typescript
// Before:
const conversations = await api.getNegotiationConversations(negotiationId);

// After:
const conversations = await api.getNegotiationConversations(negotiationId, userId);
```

## Database State

### Current Data
```
Users:
- admin (id: 20cdc4d7-2f8b-41f8-9a36-ca49d31113ac)
- moneill (id: bbeffe81-fdc1-4dff-b22d-4a79d6b9dc8e)

Negotiations:
- "General Negotiations" (admin, id: b0891b1f-e3a9-4c4a-a979-286986eda019)
- "General Negotiations" (moneill, id: c085cd9d-a584-493e-8f63-0741b313a45a)

Conversations:
- "Test Conversation" (admin's negotiation, id: 69fe9cc0-5014-4ea0-b277-7e9c9cda7b57)
```

## Testing Steps

### How to Verify Fixes

1. **Refresh browser** to load updated code (Vite hot-reload should apply changes automatically)

2. **Check Negotiation Selector Appears**:
   - Look for "ACTIVE NEGOTIATION" dropdown in sidebar
   - Should show above Model Selector
   - Should display "General Negotiations" as an option

3. **Select a Negotiation**:
   - Click dropdown and select "General Negotiations"
   - Open browser console (F12)
   - Should see: `Negotiation selected: b0891b1f-e3a9-4c4a-a979-286986eda019`
   - Should see: `Loading conversations for negotiation: ...`

4. **Verify Conversation Loads**:
   - "NEGOTIATION SESSIONS" section should show "Test Conversation"
   - "New Conversation" button should be enabled (blue, not grayed out)

5. **Test Chat Functionality**:
   - Click "Test Conversation" to select it
   - Type a message in chat input
   - Press Enter or click Send
   - Message should be sent to backend

### Expected Console Output (After All Fixes)
When you select "General Negotiations" from dropdown:
```
Negotiation dropdown changed to: b0891b1f-e3a9-4c4a-a979-286986eda019
Available negotiations: [{id: "b0891b1f-...", title: "General Negotiations", ...}]
After setCurrentNegotiation, currentNegotiationId should be: b0891b1f-e3a9-4c4a-a979-286986eda019
Current Negotiation ID changed: b0891b1f-e3a9-4c4a-a979-286986eda019
Current Negotiation object: {id: "b0891b1f-...", title: "General Negotiations", ...}
Negotiation changed: b0891b1f-e3a9-4c4a-a979-286986eda019
Loading conversations for negotiation: b0891b1f-e3a9-4c4a-a979-286986eda019
```

Key indicators of success:
- ✅ "Current Negotiation ID changed" appears (confirms state update)
- ✅ "Loading conversations for negotiation" appears (confirms API call triggered)
- ✅ Dropdown shows selected value after selection
- ✅ "New Conversation" button becomes enabled (blue, not grayed out)

## API Endpoints Verified

### Working Endpoints
```bash
# Get negotiations for user
GET /api/negotiations?user_id=20cdc4d7-2f8b-41f8-9a36-ca49d31113ac
# Returns: [{ title: "General Negotiations", id: "...", ... }]

# Get conversations for negotiation (now with userId)
GET /api/conversations/negotiation/b0891b1f-e3a9-4c4a-a979-286986eda019?user_id=20cdc4d7-2f8b-41f8-9a36-ca49d31113ac
# Returns: [{ title: "Test Conversation", id: "...", ... }]

# Health check
GET /api/health
# Returns: { status: "healthy", ... }
```

## Docker Status
- Frontend: Running on port 5173 (Vite dev server with HMR enabled)
- Backend: Running on port 8000 (FastAPI, healthy)
- PostgreSQL: Running on port 5432 (healthy)

All containers verified running via `docker ps` at time of fix.

## Follow-Up Fix: Negotiation Selection Not Persisting

### Additional Issue Discovered
After initial fixes, negotiation dropdown didn't save selection - selecting "General Negotiations" wouldn't "stick".

### Root Cause
Zustand getter pattern doesn't trigger React re-renders properly:
```typescript
// PROBLEM: getter doesn't cause component updates
get currentNegotiation() {
  const { negotiations, currentNegotiationId } = get();
  return negotiations.find(n => n.id === currentNegotiationId) || null;
}
```

When `currentNegotiationId` changes, components subscribing to `currentNegotiation` getter don't re-render.

### Solution
Compute `currentNegotiation` inside the component by subscribing to the actual state values:

**File**: `frontend/src/components/Sidebar.tsx` (line 20-28)
```typescript
// Before:
const { currentNegotiation, negotiations, loadNegotiations, setCurrentNegotiation } = useNegotiationStore();

// After:
const { negotiations, loadNegotiations, setCurrentNegotiation, currentNegotiationId } = useNegotiationStore();
const currentNegotiation = negotiations.find(n => n.id === currentNegotiationId) || null;

// Debug logging
useEffect(() => {
  console.log('Current Negotiation ID changed:', currentNegotiationId);
  console.log('Current Negotiation object:', currentNegotiation);
}, [currentNegotiationId, currentNegotiation]);
```

**Enhanced onChange handler** (line 120-126):
```typescript
onChange={(e) => {
  const selectedId = e.target.value;
  console.log('Negotiation dropdown changed to:', selectedId);
  console.log('Available negotiations:', negotiations);
  setCurrentNegotiation(selectedId);
  console.log('After setCurrentNegotiation, currentNegotiationId should be:', selectedId);
}}
```

## Files Modified
1. `frontend/src/components/Sidebar.tsx` - Added negotiation selector UI, fixed Zustand getter issue, enhanced debug logging
2. `frontend/src/components/ChatContainer.tsx` - Added user feedback for missing conversation
3. `frontend/src/services/api.ts` - Added userId parameter to getNegotiationConversations
4. `frontend/src/store/chatStore.ts` - Pass userId to API call

## Additional Notes

### Type Definitions
The `Negotiation` interface uses `title` as the primary field with `name` as a legacy alias:
```typescript
// frontend/src/types/negotiation.ts
export interface Negotiation {
  title: string;        // Primary field
  name?: string;        // Legacy alias
  // ...
}
```

### Zustand Store Pattern
The negotiation store uses Zustand with persistence:
```typescript
const { currentNegotiation, setCurrentNegotiation } = useNegotiationStore();
// currentNegotiation is a computed property
// setCurrentNegotiation(id) updates localStorage and triggers effects
```

### Effect Dependencies
The conversation loading effect depends on:
- `currentNegotiation?.id` - triggers when negotiation changes
- `user?.id` - ensures user is logged in

## Remaining Work
If chat still doesn't work after these fixes:
1. Check browser console for any new errors
2. Verify JWT token is present in localStorage
3. Check network tab for failed API calls
4. Verify conversation is actually selected (currentSessionId is set)
5. Check backend logs: `docker logs negotiator-pro-backend`

## Quick Commands for Debugging
```bash
# Check Docker containers
docker ps

# View frontend logs
docker logs --tail 50 negotiator-pro-frontend

# View backend logs
docker logs --tail 50 negotiator-pro-backend

# Check database state
docker exec negotiator-pro-backend python -c "
import asyncio, sys
sys.path.insert(0, '/app')
from backend.database import Database
async def check():
    db = Database()
    await db.connect()
    convs = await db.fetch('SELECT id, title, negotiation_id FROM conversations')
    print(convs)
    await db.disconnect()
asyncio.run(check())
"

# Test API directly
curl -s 'http://localhost:8000/api/health'
```
