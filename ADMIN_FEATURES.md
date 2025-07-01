# Admin Panel Features

## Overview
The enhanced negotiation advisor now includes a comprehensive admin panel with password protection and full system management capabilities.

## Access
- **Main Interface**: Negotiation Advisor tab (unchanged user experience)
- **Admin Panel**: Admin Panel tab (password protected)
- **Default Password**: `admin123` (change immediately after first login)

## Admin Features

### 🔐 Authentication
- **Password Protection**: Modal popup requiring admin password
- **Session Management**: 24-hour browser sessions (configurable)
- **Auto-cleanup**: Expired sessions automatically removed

### ⚙️ System Configuration
- **System Prompt Management**: 
  - Edit the core AI system prompt
  - Live reload without restart
  - Backup/restore functionality
- **Default User Prompt**: Set default prompts for users

### 📄 Document Management
- **File Upload Support**:
  - PDF documents
  - Text files (.txt)
  - Word documents (.docx)
  - Legacy Word (.doc) - if unstructured package available
- **File Validation**:
  - Size limits (50MB default)
  - Format verification
  - Content validation
- **Document Library**:
  - View all uploaded documents
  - File size and metadata display
  - Document deletion capabilities

### 🔄 Vector Database Management
- **Regenerate Vectorstore**: Rebuild from current documents
- **Automatic Processing**: New uploads trigger rebuilds
- **Performance Monitoring**: Track processing times

### 📊 Usage Statistics
- **API Usage Tracking**:
  - Token consumption by model
  - Request counts
  - Cost tracking (placeholder)
- **Historical Data**: 30-day usage summaries
- **Model Breakdown**: gpt-4o-mini vs o3-mini usage

### 🔧 Admin Settings
- **Password Management**: Change admin password securely
- **Session Configuration**: Adjust session duration
- **System Health**: Monitor application status

## File Locations

### Configuration Files
- `admin_config.json` - Admin settings and prompts
- `admin_sessions.json` - Active admin sessions
- `usage_stats.json` - API usage statistics

### Directories
- `sources/` - RAG document library
- `uploads/` - Temporary upload staging
- `vectorstore/` - FAISS embeddings database

## Security Features

### Password Security
- SHA256 hashed passwords
- No plaintext storage
- Session-based authentication

### File Upload Security
- Extension validation
- Size limits
- Content type verification
- Isolated upload directory

### Session Management
- UUID-based session tokens
- Automatic expiration
- Browser-specific sessions

## Usage Examples

### Adding New Documents
1. Go to Admin Panel → Documents tab
2. Click "Upload Documents"
3. Select PDF/TXT/DOCX files
4. Click "Upload"
5. Click "Regenerate Vector Database"

### Updating System Prompt
1. Go to Admin Panel → System Config tab
2. Click "Load Current" to see existing prompt
3. Edit the system prompt
4. Click "Save"
5. Changes take effect immediately

### Monitoring Usage
1. Go to Admin Panel → Usage Stats tab
2. Click "Refresh Stats"
3. View token usage, costs, and request counts
4. Monitor model usage patterns

## Technical Notes

### Model Configuration
- Built-in middleware handles model-specific parameters
- Automatic parameter filtering for unsupported options
- Seamless switching between models

### Performance
- Vectorstore persistence for fast startup
- Incremental document processing
- Session cleanup for memory management

### Extensibility
- Modular admin components
- Easy to add new file formats
- Configurable limits and settings

## Troubleshooting

### Common Issues
1. **Upload Fails**: Check file size and format
2. **Vectorstore Error**: Ensure documents are valid
3. **Session Expired**: Re-authenticate in admin panel
4. **Permission Error**: Check file system permissions

### Logs
- All admin actions are logged
- Check console for detailed error messages
- Log files include timing and performance data