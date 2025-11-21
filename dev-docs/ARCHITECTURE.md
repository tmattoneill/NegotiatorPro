# NegotiatorPro Architecture

## Docker-First Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          HOST MACHINE                                │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    DOCKER NETWORK                             │  │
│  │                   (negotiator-network)                        │  │
│  │                                                               │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │  │
│  │  │  PostgreSQL   │  │   Backend     │  │   Frontend    │   │  │
│  │  │   Container   │  │   Container   │  │   Container   │   │  │
│  │  │               │  │               │  │               │   │  │
│  │  │ postgres:15   │  │ Python 3.11   │  │  Node 20      │   │  │
│  │  │   -alpine     │  │   FastAPI     │  │  React+Vite   │   │  │
│  │  │               │  │   asyncpg     │  │               │   │  │
│  │  │               │  │               │  │               │   │  │
│  │  │ Port: 5432    │  │ Port: 8000    │  │ Port: 5173    │   │  │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘   │  │
│  │          │                  │                  │            │  │
│  └──────────┼──────────────────┼──────────────────┼────────────┘  │
│             │                  │                  │                │
│             ▼                  ▼                  ▼                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      DATA MOUNTS                              │  │
│  │                   (./data/ directory)                         │  │
│  │                                                               │  │
│  │  ┌─────────┐  ┌───────────┐  ┌─────────┐  ┌─────────┐      │  │
│  │  │   db/   │  │vectorstore│  │ sources/│  │ config/ │      │  │
│  │  │         │  │           │  │         │  │         │      │  │
│  │  │ PG Data │  │   FAISS   │  │  PDFs   │  │  JSON   │      │  │
│  │  │ 100MB+  │  │   10MB+   │  │  varies │  │  10KB   │      │  │
│  │  └─────────┘  └───────────┘  └─────────┘  └─────────┘      │  │
│  │                                                               │  │
│  │  ┌─────────┐  ┌───────────┐                                 │  │
│  │  │uploads/ │  │ backups/  │                                 │  │
│  │  │         │  │           │                                 │  │
│  │  │  Temp   │  │  Backups  │                                 │  │
│  │  │  Files  │  │  Storage  │                                 │  │
│  │  └─────────┘  └───────────┘                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    CODE MOUNTS (Dev Mode)                     │  │
│  │                                                               │  │
│  │  backend/ ──→ /app/backend   (hot reload)                    │  │
│  │  frontend/ ──→ /app/frontend (hot reload)                    │  │
│  │  migrations/ ──→ /docker-entrypoint-initdb.d                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Request Flow

```
User Browser (localhost:5173)
    │
    │ HTTP Request
    ▼
React Frontend Container
    │
    │ API Call (http://backend:8000)
    ▼
FastAPI Backend Container
    │
    ├──→ User Profile Operations
    │    │
    │    │ asyncpg queries
    │    ▼
    │    PostgreSQL Container
    │    │
    │    │ Read/Write
    │    ▼
    │    /data/db/ (persistent)
    │
    ├──→ RAG Operations
    │    │
    │    │ Load embeddings
    │    ▼
    │    /data/vectorstore/ (FAISS index)
    │    │
    │    │ Query LLM
    │    ▼
    │    OpenAI/Anthropic/Ollama API
    │
    └──→ Configuration
         │
         │ Read/Write JSON
         ▼
         /data/config/
```

## Database Schema

```
┌─────────────────────────────────────────────────────────────┐
│                      PostgreSQL Database                     │
│                    (negotiatorpro)                           │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐│
│  │    users     │────▶│   sessions   │────▶│chat_messages││
│  │              │     │              │     │             ││
│  │ - id (UUID)  │     │ - id (UUID)  │     │ - id        ││
│  │ - username   │     │ - user_id FK │     │ - session_id││
│  │ - email      │     │ - expires_at │     │ - content   ││
│  │ - first_name │     │ - role       │     │ - model     ││
│  │ - last_name  │     └──────────────┘     └─────────────┘│
│  │ - openai_key │                                          │
│  │ - anthro_key │     ┌──────────────┐                    │
│  │ - password   │────▶│ usage_logs   │                    │
│  │ - role       │     │              │                    │
│  └──────────────┘     │ - timestamp  │                    │
│                       │ - user_id FK │                    │
│  ┌──────────────┐     │ - model      │                    │
│  │  documents   │     │ - tokens     │                    │
│  │              │     │ - cost       │                    │
│  │ - id         │     └──────────────┘                    │
│  │ - filename   │                                          │
│  │ - file_hash  │     ┌──────────────┐                    │
│  │ - uploaded_by│────▶│  llm_config  │                    │
│  └──────────────┘     │              │                    │
│                       │ - backend    │                    │
│                       │ - model      │                    │
│                       │ - is_active  │                    │
│                       └──────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## User Profile Encryption Flow

```
┌──────────────┐
│  User Input  │
│              │
│ API Key:     │
│ sk-proj-XYZ  │
└──────┬───────┘
       │
       │ POST /api/users/
       ▼
┌──────────────────────┐
│ EncryptionManager    │
│                      │
│ Key: ENCRYPTION_KEY  │
│ (from .env)          │
└──────┬───────────────┘
       │
       │ Fernet.encrypt()
       ▼
┌──────────────────────┐
│ Encrypted in DB:     │
│                      │
│ gAAAAABh2Xk...      │
│ (base64 encoded)     │
└──────┬───────────────┘
       │
       │ Storage in PostgreSQL
       ▼
┌──────────────────────┐
│ users.openai_api_key │
│ (TEXT column)        │
└──────────────────────┘

Retrieval (reverse):
GET /api/users/{id}/api-keys
   → Query DB → Fernet.decrypt() → Return plaintext
```

## File System Layout

```
Host: ./data/
├── db/                         # PostgreSQL data directory
│   ├── base/                   # Database files
│   ├── global/                 # Cluster-wide data
│   ├── pg_wal/                 # Write-ahead log
│   └── postgresql.conf         # Config
│
├── vectorstore/                # FAISS embeddings
│   ├── index.faiss             # Vector index
│   ├── index.pkl               # Metadata
│   └── embedding_metadata.json # Config
│
├── sources/                    # Source documents
│   ├── getting_to_yes.pdf
│   ├── never_split.pdf
│   └── ...
│
├── config/                     # Application config
│   ├── admin_config.json
│   ├── llm_backend_config.json
│   ├── usage_stats.json
│   └── ...
│
├── uploads/                    # Temporary uploads
│   └── (cleaned after processing)
│
└── backups/                    # Backup storage
    ├── db_20250120.sql
    └── config_20250120/

Container Mapping:
./data/db → /var/lib/postgresql/data (postgres)
./data/vectorstore → /app/vectorstore (backend)
./data/sources → /app/sources (backend)
./data/config → /app/config (backend)
```

## Technology Stack

### Backend Container
- **Base**: Python 3.11
- **Web Framework**: FastAPI
- **Database**: asyncpg (PostgreSQL async driver)
- **Encryption**: cryptography (Fernet)
- **Auth**: passlib (bcrypt)
- **RAG**: LangChain, FAISS, OpenAI embeddings
- **LLM**: OpenAI, Anthropic, Ollama clients

### Database Container
- **Database**: PostgreSQL 15 (Alpine)
- **Features**: JSONB, UUID, triggers, views

### Frontend Container
- **Runtime**: Node.js 20 (Alpine)
- **Framework**: React 18
- **Build Tool**: Vite
- **State**: Zustand
- **HTTP**: Axios

## Security Layers

```
┌────────────────────────────────────────────┐
│         Application Security               │
├────────────────────────────────────────────┤
│ 1. Password Hashing (bcrypt)               │
│    - Auto-salting                          │
│    - Configurable work factor              │
├────────────────────────────────────────────┤
│ 2. API Key Encryption (Fernet)             │
│    - Symmetric encryption                  │
│    - Environment-based key                 │
├────────────────────────────────────────────┤
│ 3. SQL Injection Prevention                │
│    - Parameterized queries (asyncpg)       │
│    - No string concatenation               │
├────────────────────────────────────────────┤
│ 4. Input Validation (Pydantic)             │
│    - Type checking                         │
│    - Length constraints                    │
│    - Email validation                      │
├────────────────────────────────────────────┤
│ 5. Docker Isolation                        │
│    - Network isolation                     │
│    - Non-root users                        │
│    - Resource limits                       │
└────────────────────────────────────────────┘
```

## Deployment Workflow

```
Development                Production
├─ docker-compose.yml     ├─ docker-compose.prod.yml
├─ Hot reload enabled     ├─ Optimized builds
├─ Debug logging          ├─ Production logging
├─ Port 5173 exposed      ├─ Behind Nginx/Traefik
└─ Local .env             └─ Secrets management

Common:
├─ PostgreSQL container
├─ /data mount
├─ Auto-migrations
└─ Health checks
```

---

This architecture provides:
- ✅ Complete containerization
- ✅ Clean data separation
- ✅ Production-ready security
- ✅ Easy backup/restore
- ✅ Horizontal scalability
- ✅ Development-friendly hot reload
