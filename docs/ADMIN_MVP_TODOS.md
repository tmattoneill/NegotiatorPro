# Admin MVP - Priority TODOs

## CRITICAL (Security)

### 1. Protect System Prompt Endpoints
**File:** `backend/api/routes/admin.py`

The following endpoints are UNPROTECTED and accessible to any authenticated user:
- `GET /api/admin/system-prompt`
- `PUT /api/admin/system-prompt`
- `GET /api/admin/system-prompt/backups`
- `GET /api/admin/system-prompt/backup/{filename}`
- `POST /api/admin/system-prompt/restore/{filename}`

**Fix:** Add `admin: Dict = Depends(verify_admin)` parameter to each endpoint function.

### 2. Add Role Check to SystemPromptEditor
**File:** `frontend/src/components/SystemPromptEditor.tsx`

The component doesn't verify admin role directly. While sidebar gates access, the component should have its own check.

**Fix:** Add same pattern as `AdminPanel.tsx`:
```tsx
const user = useAuthStore((state) => state.user);
if (user?.role !== 'admin') {
  return <div>Access Denied</div>;
}
```

---

## HIGH PRIORITY (Functionality)

### 3. Test Admin Password Login Flow
The new `/admin-login` route uses the legacy `/api/auth/login` endpoint. Verify:
- Password validation works
- JWT token is properly generated
- Admin panel access works after login

### 4. Verify LLM Config Persistence
Test that model selection changes via Admin Panel actually persist:
- Change default model → verify chat uses new model
- Change premium model → verify premium toggle uses new model
- Enable/disable backend → verify availability changes

---

## MEDIUM PRIORITY (Polish)

### 5. Remove Debug Console Logs
**File:** `frontend/src/components/Sidebar.tsx`

Remove or wrap in dev-only check:
- Lines 28-30 (negotiation ID logging)
- Lines 55-58 (conversation loading logging)
- Lines 215-218 (dropdown change logging)

### 6. Improve Error Messages in LLMConfigTab
**File:** `frontend/src/components/LLMConfigTab.tsx`

Add more specific error messages for:
- Backend has no API key configured
- Model selection failed
- Network errors

### 7. Add Loading States to Admin Login
**File:** `frontend/src/components/AdminLogin.tsx`

- Add loading spinner during auth
- Disable form while submitting
- Show success feedback before redirect

---

## LOW PRIORITY (Nice-to-have)

### 8. Add Refresh Button to LLM Config Tab
Allow admin to refresh Ollama model list without page reload.

### 9. Add Model Cost Information
Display cost-per-token info in model selection modal.

### 10. Add Session Timeout Warning
Show warning when admin session is about to expire.

---

## Testing Checklist

Before considering admin MVP complete:

- [ ] Admin can login via `/admin-login` with password
- [ ] Admin can login via `/login` with admin user account
- [ ] Non-admin users cannot access admin panel
- [ ] Non-admin users cannot access system prompt endpoints
- [ ] LLM config changes persist and affect chat
- [ ] All admin tabs work (Users, Negotiations, Usage, LLM Config, Database)
- [ ] Database reset works with confirmation
- [ ] System prompt edit/backup/restore works
