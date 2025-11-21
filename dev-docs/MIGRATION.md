# NegotiatorPro Data Model Migration

This document tracks the migration from a simple chat application to a comprehensive multi-user, multi-negotiation platform with isolated contexts.

## Migration Status

### ✅ Phase 1: Core Negotiations Entity (COMPLETED)

**Database:**
- ✅ Created `negotiations` table (migration 003)
  - Stores negotiation metadata (name, description, background)
  - Links to users via `user_id`
  - Supports key players (JSONB array)
  - Model preferences (preferred_backend, preferred_model)
  - Status tracking (active, archived, completed)
  - Visibility (private/company - company to be implemented in Phase 2)
- ✅ Automatic creation of default "General Negotiations" for existing users
- ✅ Indexes on user_id, status, created_at, updated_at
- ✅ Auto-update trigger for updated_at timestamp

**Backend:**
- ✅ Created `backend/negotiation_manager.py`
  - `NegotiationManager` class with full CRUD operations
  - `Negotiation`, `NegotiationCreate`, `NegotiationUpdate` Pydantic models
  - `KeyPlayer` model for negotiation participants
  - User authorization checks on all operations
- ✅ Created `backend/api/routes/negotiations.py`
  - POST /api/negotiations/ - Create negotiation
  - GET /api/negotiations/{id} - Get single negotiation
  - GET /api/negotiations/ - List user's negotiations
  - PUT/PATCH /api/negotiations/{id} - Update negotiation
  - DELETE /api/negotiations/{id} - Delete negotiation
- ✅ Registered negotiations router in main.py

**Frontend:**
- ✅ Created `frontend/src/types/negotiation.ts`
  - TypeScript interfaces matching backend models
- ✅ Created `frontend/src/store/negotiationStore.ts`
  - Zustand store with persistence
  - Full CRUD operations via API
  - Current negotiation selection
  - Error handling and loading states

**Files Created:**
- `/migrations/003_create_negotiations.sql`
- `/backend/negotiation_manager.py`
- `/backend/api/routes/negotiations.py`
- `/frontend/src/types/negotiation.ts`
- `/frontend/src/store/negotiationStore.ts`

**Files Modified:**
- `/backend/api/routes/__init__.py` - Added negotiations_router export
- `/backend/api/main.py` - Registered negotiations_router

---

## 🚧 Phase 2: Conversations & Message Persistence (NEXT)

### Database Changes Required

**Migration 004: Create conversations table**
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    negotiation_id UUID REFERENCES negotiations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    last_message_at TIMESTAMP,

    CONSTRAINT conversations_unique_negotiation_user UNIQUE (negotiation_id, id)
);

-- Indexes
CREATE INDEX idx_conversations_negotiation_id ON conversations(negotiation_id);
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_updated_at ON conversations(updated_at DESC);

-- Trigger for updated_at
CREATE TRIGGER trigger_update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_negotiations_updated_at();  -- Reuse function
```

**Migration 004: Modify chat_messages table**
```sql
-- Add new foreign keys
ALTER TABLE chat_messages ADD COLUMN conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE;
ALTER TABLE chat_messages ADD COLUMN negotiation_id UUID REFERENCES negotiations(id) ON DELETE CASCADE;

-- Create indexes
CREATE INDEX idx_chat_messages_conversation_id ON chat_messages(conversation_id);
CREATE INDEX idx_chat_messages_negotiation_id ON chat_messages(negotiation_id);

-- Migrate existing messages to default "General Negotiations"
-- This will link all existing chat_messages to the auto-created General negotiation for each user
UPDATE chat_messages cm
SET negotiation_id = n.id
FROM negotiations n
WHERE n.user_id = cm.user_id
  AND n.name = 'General Negotiations'
  AND cm.negotiation_id IS NULL;

-- Note: conversation_id will remain NULL for now until we create conversations
-- We can optionally create a default conversation per user's general negotiation
```

### Backend Implementation

**Files to Create:**
1. `/backend/conversation_manager.py`
   - `ConversationManager` class
   - `Conversation`, `ConversationCreate`, `ConversationUpdate` models
   - `ConversationWithMessages` model (includes message list)
   - CRUD operations with negotiation/user authorization

2. `/backend/api/routes/conversations.py`
   - POST /api/negotiations/{negotiation_id}/conversations - Create conversation
   - GET /api/negotiations/{negotiation_id}/conversations - List conversations
   - GET /api/conversations/{id} - Get conversation with messages
   - PUT/PATCH /api/conversations/{id} - Update conversation (title, etc.)
   - DELETE /api/conversations/{id} - Delete conversation
   - POST /api/conversations/{id}/messages - Add message to conversation

**Files to Modify:**
1. `/backend/api/routes/chat.py`
   - Modify POST /api/chat to accept conversation_id and negotiation_id
   - Save messages to database via conversation_id
   - Link to current negotiation context
   - Return conversation_id in response

2. `/backend/api/main.py`
   - Register conversations_router

3. `/backend/api/routes/__init__.py`
   - Export conversations_router

### Frontend Implementation

**Files to Create:**
1. `/frontend/src/types/conversation.ts`
   ```typescript
   export interface Conversation {
     id: string;
     negotiation_id: string;
     user_id: string;
     title: string;
     created_at: string;
     updated_at: string;
     message_count: number;
     last_message_at?: string;
   }

   export interface ConversationCreate {
     title: string;
     negotiation_id: string;
   }
   ```

2. `/frontend/src/store/conversationStore.ts`
   - Similar to negotiationStore but for conversations
   - Link to current negotiation
   - Load messages on conversation selection

**Files to Modify:**
1. `/frontend/src/store/chatStore.ts`
   - Connect to conversationStore
   - Persist sessions as conversations via API
   - Load conversation history from database
   - Track current conversation_id and negotiation_id

2. `/frontend/src/components/Sidebar.tsx`
   - Show conversations grouped by negotiation
   - "New Conversation" button creates DB record
   - Switch between conversations loads from API

3. `/frontend/src/components/ChatContainer.tsx`
   - Load messages from conversation on mount
   - Include conversation_id in all API calls

### Migration of Existing Data

**Strategy:** Create default conversations for existing chat_messages
- Group existing messages by user and session_id
- Create one conversation per session
- Link messages to conversations
- All under "General Negotiations" negotiation

---

## 📋 Phase 3: Documents Per Negotiation (TODO)

### Database Changes Required

**Migration 005: Link documents to negotiations**
```sql
-- Add negotiation_id to documents table
ALTER TABLE documents ADD COLUMN negotiation_id UUID REFERENCES negotiations(id) ON DELETE CASCADE;
ALTER TABLE documents ADD COLUMN is_shared BOOLEAN DEFAULT false;

CREATE INDEX idx_documents_negotiation_id ON documents(negotiation_id);
CREATE INDEX idx_documents_shared ON documents(is_shared);

-- Mark existing documents as shared (global)
UPDATE documents SET is_shared = true WHERE negotiation_id IS NULL;
```

### Backend Implementation

**Files to Modify:**
1. `/backend/document_manager.py`
   - Add `negotiation_id` parameter to upload methods
   - Support `is_shared` flag for global vs per-negotiation docs
   - Add XLS/CSV support using pandas or openpyxl

2. `/backend/rag_engine.py`
   - Implement hybrid RAG strategy:
     - Global vectorstore for shared documents (is_shared=true)
     - Per-negotiation filtering using metadata
   - Add `negotiation_id` to document metadata when indexing
   - Filter search results by negotiation context

**Files to Create:**
1. `/backend/api/routes/documents.py`
   - POST /api/negotiations/{id}/documents - Upload document
   - GET /api/negotiations/{id}/documents - List documents
   - DELETE /api/negotiations/{id}/documents/{doc_id} - Remove document
   - GET /api/documents/shared - List shared/global documents

### Frontend Implementation

**Files to Create:**
1. `/frontend/src/components/DocumentUpload.tsx`
   - File upload UI within negotiation view
   - Support PDF, TXT, DOCX, DOC, XLS, XLSX, CSV
   - Upload progress indicator
   - File list with delete option

2. `/frontend/src/types/document.ts`
   ```typescript
   export interface Document {
     id: string;
     negotiation_id?: string;
     filename: string;
     file_type: string;
     file_size: number;
     is_shared: boolean;
     upload_date: string;
     is_processed: boolean;
   }
   ```

**Files to Modify:**
1. Create negotiation detail view showing documents
2. Add document upload button to negotiation view

---

## 🌐 Phase 4: URLs & Web Content (TODO)

### Database Changes Required

**Migration 006: Create negotiation_urls table**
```sql
CREATE TABLE negotiation_urls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    negotiation_id UUID REFERENCES negotiations(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title VARCHAR(500),
    content TEXT,
    scraped_at TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_status CHECK (status IN ('pending', 'scraped', 'failed'))
);

CREATE INDEX idx_negotiation_urls_negotiation_id ON negotiation_urls(negotiation_id);
CREATE INDEX idx_negotiation_urls_status ON negotiation_urls(status);
```

### Backend Implementation

**Files to Create:**
1. `/backend/url_scraper.py`
   - URL validation and normalization
   - Web scraping using BeautifulSoup or Playwright
   - Content extraction and cleaning
   - Error handling for failed scrapes
   - Optional: Background task queue (Celery/Redis)

2. `/backend/api/routes/urls.py`
   - POST /api/negotiations/{id}/urls - Add URL
   - GET /api/negotiations/{id}/urls - List URLs
   - POST /api/negotiations/{id}/urls/{url_id}/scrape - Trigger scraping
   - DELETE /api/negotiations/{id}/urls/{url_id} - Remove URL

**Files to Modify:**
1. `/backend/rag_engine.py`
   - Index scraped URL content into vectorstore
   - Link to negotiation context via metadata

### Frontend Implementation

**Files to Create:**
1. `/frontend/src/components/URLManager.tsx`
   - URL input form
   - URL list with scraping status badges
   - Retry failed scrapes
   - Preview scraped content

2. `/frontend/src/types/url.ts`
   ```typescript
   export interface NegotiationURL {
     id: string;
     negotiation_id: string;
     url: string;
     title?: string;
     status: 'pending' | 'scraped' | 'failed';
     scraped_at?: string;
     error_message?: string;
   }
   ```

---

## ⚙️ Phase 5: User Preferences & Model Selection (TODO)

### Database Changes Required

**Migration 007: Add user preferences**
```sql
ALTER TABLE users ADD COLUMN preferred_backend VARCHAR(50);
ALTER TABLE users ADD COLUMN preferred_default_model VARCHAR(100);
ALTER TABLE users ADD COLUMN preferred_premium_model VARCHAR(100);

-- Set defaults from current settings if available
-- (Could pull from current frontend state or leave NULL for user to set)
```

### Backend Implementation

**Files to Modify:**
1. `/backend/user_profile.py`
   - Add preferred_backend, preferred_default_model, preferred_premium_model to UserProfile model
   - Add to UserProfileCreate and UserProfileUpdate models

2. `/backend/api/routes/chat.py`
   - Use user preferences as default model selection
   - Allow negotiation-level override
   - Fallback chain: Negotiation → User → System default

### Frontend Implementation

**Files to Modify:**
1. `/frontend/src/components/UserProfile.tsx`
   - Add model preference selection fields
   - Show current default/premium models

2. `/frontend/src/store/authStore.ts`
   - Add preferred model fields to User interface

---

## 🎯 Current Status Summary

**What Works:**
- ✅ Full user management (profiles, API keys, authentication)
- ✅ Negotiations CRUD (create, read, update, delete)
- ✅ Per-negotiation model preferences
- ✅ Key players tracking
- ✅ Chat messaging (in-memory, not persisted)
- ✅ Document table structure
- ✅ Model selection (per-request)

**What's Missing:**
- ❌ Conversations persistence (messages disappear on refresh)
- ❌ Negotiation → Conversation linkage
- ❌ Negotiation → Document linkage
- ❌ Per-negotiation document isolation
- ❌ URL scraping and storage
- ❌ User-level model preferences
- ❌ Frontend UI for negotiations (still shows old sessions)

**Next Immediate Steps:**
1. Update Sidebar to show negotiations instead of sessions
2. Create "New Negotiation" dialog component
3. Load user's negotiations on app mount
4. Then proceed with Phase 2 (Conversations)

---

## 📝 Implementation Notes

### Design Decisions Made

**RAG Strategy: Hybrid Approach**
- Global vectorstore for shared documents (is_shared=true)
- Per-negotiation filtering using metadata
- Balances isolation with performance

**Data Migration: Create Default Negotiation**
- All existing users get "General Negotiations" auto-created
- Existing messages will link to this default negotiation
- Preserves all historical data

**Company Visibility: Phase 2 Feature**
- Start with Private only
- Company sharing requires additional access control logic
- Will be implemented after core features are stable

### Testing Checklist for Phase 1

- [ ] Can create negotiation via API
- [ ] Can list user's negotiations
- [ ] Can update negotiation details
- [ ] Can delete negotiation
- [ ] Authorization prevents access to other users' negotiations
- [ ] Default "General Negotiations" exists for all users
- [ ] Frontend store can load negotiations
- [ ] Frontend store persists selection to localStorage

### Known Issues

1. Frontend still shows old "sessions" UI - needs update to negotiations
2. Chat messages don't persist - Phase 2 will fix
3. Model preferences not yet flowing through to chat - Phase 5
4. No UI for adding key players yet - will add with negotiation detail view

---

## 🔄 Quick Reference: Migration Files

| Migration | Purpose | Status |
|-----------|---------|--------|
| 001_initial_schema.sql | Users, documents, chat_messages tables | ✅ Complete |
| 002_add_user_profile_fields.sql | First name, last name, API keys | ✅ Complete |
| 003_create_negotiations.sql | Negotiations table | ✅ Complete |
| 004_create_conversations.sql | Conversations and link to messages | 🚧 Next |
| 005_link_documents_negotiations.sql | Add negotiation_id to documents | 📋 TODO |
| 006_create_negotiation_urls.sql | URL storage and scraping | 📋 TODO |
| 007_add_user_preferences.sql | User model preferences | 📋 TODO |

---

## 🚀 How to Continue from Here

1. **Test Phase 1 Backend:**
   ```bash
   # Test negotiations API
   curl -X POST "http://localhost:8000/api/negotiations/?user_id=YOUR_USER_ID" \
     -H "Content-Type: application/json" \
     -d '{"name": "Test Negotiation", "description": "Testing API"}'
   ```

2. **Next: Update Frontend UI**
   - Modify Sidebar.tsx to show negotiations
   - Create NewNegotiationDialog.tsx
   - Load negotiations on app mount using useNegotiationStore
   - Test negotiation selection and switching

3. **Then: Start Phase 2**
   - Create migration 004
   - Implement ConversationManager
   - Connect chat messages to conversations
   - Persist conversation history

---

*Last Updated: 2025-01-20*
*Current Phase: Phase 1 Complete, UI Update Needed*
