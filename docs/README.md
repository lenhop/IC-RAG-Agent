# UDS Agent Documentation

**Version:** 1.0.0  
**Last Updated:** 2026-03-06

---

## Quick Navigation

### For Users
Start here if you want to use the UDS Agent:
- **[User Guide](guides/UDS_USER_GUIDE.md)** - 50+ query examples, best practices, FAQ
- **[Business Glossary](BUSINESS_GLOSSARY.md)** - Amazon seller terminology and table reference

### For Developers
Start here if you want to develop or extend the system:
- **[Developer Guide](guides/UDS_DEVELOPER_GUIDE.md)** - Architecture, tool development, code structure
- **[API Reference](guides/UDS_API_REFERENCE.md)** - Complete API documentation with examples
- **[HuggingFace Setup Guide](guides/HUGGINGFACE_SETUP_GUIDE.md)** - HuggingFace CLI and model download
- **[System Framework](FRAMEWORK.md)** - System architecture diagrams and component interactions

### For Operations
Start here if you're deploying or operating the system:
- **[Deployment Guide](guides/UDS_DEPLOYMENT_GUIDE.md)** - Docker, Alibaba Cloud ECS deployment
- **[Operations Manual](OPERATIONS.md)** - Daily ops, troubleshooting, incidents, launch checklist
- **[Operations Guide](guides/UDS_OPERATIONS_GUIDE.md)** - Detailed monitoring and maintenance

### For Project Management
Start here for project overview and decisions:
- **[Project Documentation](PROJECT.md)** - Complete project summary, timeline, metrics, architecture decisions

### Reference
- **[Business Glossary](BUSINESS_GLOSSARY.md)** - Amazon seller terms, metrics, table mappings
- **[API Specification](../specs/UDS_API_SPEC.yaml)** - OpenAPI spec (YAML)

---

## Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── PROJECT.md                   # Project summary, metrics, ADRs
├── OPERATIONS.md                # Operations manual (daily ops, incidents, launch)
├── BUSINESS_GLOSSARY.md         # Amazon seller terminology and table reference
│
├── guides/                      # User guides
│   ├── UDS_USER_GUIDE.md        # How to use UDS Agent
│   ├── UDS_DEVELOPER_GUIDE.md   # Development guide
│   ├── UDS_API_REFERENCE.md     # API documentation
│   ├── UDS_DEPLOYMENT_GUIDE.md  # Deployment instructions
│   ├── UDS_OPERATIONS_GUIDE.md  # Detailed operations
│   └── HUGGINGFACE_SETUP_GUIDE.md # HuggingFace CLI and models
│
└── archive/                     # Historical reference documents
    └── [reference materials]    # Architecture decisions, guides, etc.

specs/                           # At project root
├── UDS_API_SPEC.yaml            # OpenAPI spec (YAML)

tasks/
└── [All plans, summaries, reports, and task files]
```

**Note:** Plan files, summary files, status reports, and task-related documents are stored in the `tasks/` folder, not in `docs/`. The `docs/` folder contains only user-facing documentation.

---

## What's in Each Document

### User Guide
- 50+ query examples across 5 domains (sales, inventory, financial, product, BI)
- Best practices for writing queries
- FAQ and troubleshooting
- Query optimization tips

### Business Glossary
- Amazon-specific terminology (ASIN, SKU, FBA, FBM, etc.)
- Business metrics definitions (GMV, AOV, conversion rate)
- Common business questions mapped to tables
- Table quick reference

### Developer Guide
- System architecture and components
- 16 UDS tools documentation
- How to create new tools
- Code structure and conventions
- Development setup

### API Reference
- 11 REST API endpoints
- Request/response schemas
- Code examples in multiple languages
- Authentication and error handling

### Deployment Guide
- Local development setup
- Docker deployment
- Alibaba Cloud ECS deployment
- CI/CD pipeline configuration
- Environment variables

### Operations Guide
- Monitoring with Prometheus and Grafana
- Troubleshooting common issues
- Maintenance procedures
- Backup and recovery
- Performance tuning

### Operations Manual
- Daily operations checklist
- Incident response procedures
- Production launch checklist
- Quick reference commands
- Escalation procedures

### Project Documentation
- Executive summary
- Project timeline and phases
- Key metrics and achievements
- Architecture decisions (8 ADRs)
- Team contributions
- Lessons learned

---

## Getting Started

1. **New User?** Start with [User Guide](guides/UDS_USER_GUIDE.md)
2. **Developer?** Read [Developer Guide](guides/UDS_DEVELOPER_GUIDE.md)
3. **Deploying?** Follow [Deployment Guide](guides/UDS_DEPLOYMENT_GUIDE.md)
4. **Operating?** Use [Operations Manual](OPERATIONS.md)

---

## Quick Links

- **GitHub Repository:** IC-RAG-Agent
- **API Endpoint:** `http://<host>:8001/api/v1/uds/`
- **Health Check:** `http://<host>:8001/health`
- **API Docs:** `http://<host>:8001/docs` (Swagger)
- **Grafana:** `http://<host>:3000`

---

## Support

- **Documentation Issues:** Open GitHub issue
- **Operational Issues:** Follow [Operations Manual](OPERATIONS.md) incident response
- **Questions:** See FAQ in [User Guide](guides/UDS_USER_GUIDE.md)

---

**Documentation Version:** 1.0.0  
**System Version:** 1.0.0  
**Status:** Production Ready
