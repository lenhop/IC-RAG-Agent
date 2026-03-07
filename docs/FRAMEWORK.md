# IC-RAG-Agent System Framework

**Version:** 1.0.0  
**Last Updated:** 2026-03-06

This document describes the system framework for the IC-RAG-Agent project using Mermaid diagrams.

---

## 1. System Overview

IC-RAG-Agent is an **Intent Classification + Retrieval-Augmented Generation** system with three main subsystems:

- **UDS Agent** – Business Intelligence for Amazon seller data (ClickHouse)
- **RAG Pipeline** – Document retrieval and hybrid generation with intent classification
- **SP-API Agent** – Seller Operations API integration for Amazon marketplace

```mermaid
flowchart TB
    subgraph Users
        U1[Business User]
        U2[Document User]
        U3[Seller Ops User]
    end

    subgraph APIs["API Layer"]
        UDS_API[UDS API<br/>FastAPI :8000]
        RAG_API[RAG API<br/>FastAPI :8000]
        SP_API[SP-API<br/>FastAPI]
    end

    subgraph Agents["Agent Layer"]
        UDS[UDS Agent<br/>ReAct + 16 Tools]
        RAG[RAG Pipeline<br/>4 Intent Methods]
        SP[SP-API Agent<br/>ReAct + 10 Tools]
    end

    subgraph Data["Data Layer"]
        CH[(ClickHouse)]
        Chroma[(ChromaDB)]
        Redis[(Redis)]
    end

    U1 --> UDS_API
    U2 --> RAG_API
    U3 --> SP_API

    UDS_API --> UDS
    RAG_API --> RAG
    SP_API --> SP

    UDS --> CH
    UDS --> Redis
    RAG --> Chroma
    RAG --> Redis
    SP --> Chroma
```

---

## 1.1 Unified Gateway and Five Workflows

The unified gateway (port 8000) routes each query to one of five workflows:

| Workflow | Backend | Port | Note |
|----------|---------|------|------|
| General Knowledge | RAG (general mode) | 8002 | Remote LLM (e.g. DeepSeek) via RAG config. |
| Amazon Document | RAG (documents mode) | 8002 | Chroma retrieval, Amazon-docs bias. |
| Enterprise/IC Document | RAG (documents mode) or skip | 8002 | **Not ready yet:** Chroma not populated; gateway returns a friendly message when routed to IC docs. Workflow remains in diagrams. |
| SP-API Agent | SP-API Agent | 8003 | Seller operations. |
| UDS Agent | UDS Agent | 8001 | BI/analytics (ClickHouse). |

**IC docs:** Retrieval is not ready yet (Chroma not populated). The gateway still shows this workflow and returns a friendly message when the route is IC docs; set `IC_DOCS_ENABLED=true` once Chroma is populated.

---

## 2. Architecture Layers

```mermaid
flowchart TB
    subgraph Layer1["API Layer"]
        FastAPI[FastAPI]
        Uvicorn[Uvicorn]
    end

    subgraph Layer2["Agent Layer"]
        ReAct[ReAct Agent]
        Intent[Intent Classifier]
        Planner[Task Planner]
    end

    subgraph Layer3["Tool Layer"]
        UDS_Tools[UDS Tools (16)]
        SP_Tools[SP-API Tools (10)]
    end

    subgraph Layer4["Data Layer"]
        ClickHouse[ClickHouse]
        ChromaDB[ChromaDB]
        Redis[Redis]
    end

    subgraph Layer5["LLM Layer"]
        Ollama[Ollama]
        Remote[Remote API]
    end

    Layer1 --> Layer2
    Layer2 --> Layer3
    Layer3 --> Layer4
    Layer2 --> Layer5
```

---

## 3. Module Structure

```mermaid
flowchart LR
    subgraph src["src/"]
        subgraph uds["uds/"]
            api[api.py]
            agent[uds_agent.py]
            client[uds_client.py]
            intent[intent_classifier.py]
            planner[task_planner.py]
            formatter[result_formatter.py]
            cache[cache.py]
            tools[tools/]
        end

        subgraph rag["rag/"]
            pipeline[query_pipeline.py]
            intent_m[intent_methods.py]
            aggregator[intent_aggregator.py]
            chroma[chroma_loaders.py]
            embeddings[embeddings.py]
        end

        subgraph agent_mod["agent/"]
            react[react_agent.py]
            models[models.py]
        end

        subgraph sp_api["sp_api/"]
            sp_agent[sp_api_agent.py]
            sp_client[sp_api_client.py]
            sp_tools[tools/]
        end
    end

    subgraph external["external/"]
        ai_toolkit[ai-toolkit]
        llama[llama.cpp]
    end

    agent --> react
    agent --> tools
    sp_agent --> react
    sp_agent --> sp_tools
    react --> ai_toolkit
```

---

## 4. UDS Agent Flow

```mermaid
flowchart TB
    subgraph Input
        Q[User Query]
    end

    subgraph UDS["UDS Agent Pipeline"]
        IC[Intent Classifier]
        TP[Task Planner]
        RA[ReAct Agent]
        RF[Result Formatter]
    end

    subgraph Tools["Tools"]
        Schema[Schema Tools]
        Query[Query Tools]
        Analysis[Analysis Tools]
        Viz[Visualization Tools]
    end

    subgraph Data
        CH[(ClickHouse)]
        Cache[(Redis)]
    end

    Q --> IC
    IC --> TP
    TP --> RA
    RA --> Tools
    Tools --> CH
    RA --> RF
    RF --> Out[Response]

    IC -.-> Cache
    Tools -.-> Cache
```

---

## 5. UDS Intent and Tool Categories

```mermaid
flowchart TB
    subgraph Intents["Intent Domains"]
        S[Sales]
        I[Inventory]
        F[Financial]
        P[Product]
        C[Comparison]
        G[General]
    end

    subgraph SchemaTools["Schema Tools (4)"]
        LT[list_tables]
        DT[describe_table]
        RT[get_table_relationships]
        SC[search_columns]
    end

    subgraph QueryTools["Query Tools (4)"]
        GS[generate_sql]
        EQ[execute_query]
        VQ[validate_query]
        XQ[explain_query]
    end

    subgraph AnalysisTools["Analysis Tools (5)"]
        ST[analyze_sales_trend]
        IA[analyze_inventory]
        PP[analyze_product_performance]
        FS[financial_summary]
        CM[compare_metrics]
    end

    subgraph VizTools["Visualization Tools (3)"]
        CC[create_chart]
        CD[create_dashboard]
        EV[export_visualization]
    end

    Intents --> SchemaTools
    Intents --> QueryTools
    Intents --> AnalysisTools
    Intents --> VizTools
```

---

## 6. RAG Pipeline Flow

```mermaid
flowchart TB
    subgraph Input
        Query[User Query]
    end

    subgraph Preprocess
        QR[Query Rewriting]
    end

    subgraph Parallel["Four Parallel Intent Methods"]
        M1[Documents<br/>Chroma Retrieval]
        M2[Keywords<br/>Domain Match]
        M3[FAQ<br/>Similarity]
        M4[LLM<br/>Zero-shot NLI]
    end

    subgraph Aggregate
        AGG[Aggregate Yes/No]
    end

    subgraph Mode["Answer Mode"]
        GEN[General Mode]
        DOC[Documents Mode]
        HYB[Hybrid Mode]
    end

    subgraph Generation
        Local[Local LLM]
        Remote[Remote LLM]
    end

    Query --> QR
    QR --> M1
    QR --> M2
    QR --> M3
    QR --> M4
    M1 --> AGG
    M2 --> AGG
    M3 --> AGG
    M4 --> AGG
    AGG --> GEN
    AGG --> DOC
    AGG --> HYB
    GEN --> Remote
    DOC --> Local
    HYB --> Remote
    HYB --> Local
```

---

## 7. ReAct Agent Loop

```mermaid
flowchart LR
    subgraph ReAct["ReAct Loop"]
        T[Thought]
        A[Action]
        O[Observation]
    end

    T --> A
    A --> O
    O --> T

    subgraph Tools
        T1[Tool 1]
        T2[Tool 2]
        Tn[Tool N]
    end

    A --> Tools
    Tools --> O
```

---

## 8. Data Flow

```mermaid
flowchart TB
    subgraph Inbound
        U[User Request]
    end

    subgraph Processing
        API[API Layer]
        Agent[Agent]
        Tools[Tools]
    end

    subgraph Storage
        CH[(ClickHouse<br/>40M+ rows)]
        Chroma[(ChromaDB<br/>Vector Store)]
        Redis[(Redis<br/>Cache)]
    end

    subgraph Outbound
        R[Response]
    end

    U --> API
    API --> Agent
    Agent --> Tools
    Tools --> CH
    Tools --> Cache
    Agent -.-> Chroma
    Agent -.-> Redis
    Tools --> R
```

---

## 9. Deployment Architecture

```mermaid
flowchart TB
    subgraph Cloud["Alibaba Cloud ECS"]
        subgraph Containers["Docker Containers"]
            UDS[UDS Agent<br/>:8001]
            CH[ClickHouse<br/>:8123]
            R[Redis<br/>:6379]
            Chroma[ChromaDB<br/>:8000]
        end

        subgraph Monitoring["Monitoring"]
            Prom[Prometheus]
            Graf[Grafana]
        end
    end

    subgraph External
        Ollama[Ollama LLM]
        Nginx[Nginx Proxy]
    end

    Nginx --> UDS
    UDS --> CH
    UDS --> R
    UDS --> Ollama
    UDS --> Prom
    Prom --> Graf
```

---

## 10. Technology Stack

```mermaid
flowchart TB
    subgraph Framework["Framework"]
        FastAPI[FastAPI]
        Pydantic[Pydantic]
        Uvicorn[Uvicorn]
    end

    subgraph Agent["Agent & LLM"]
        LangChain[LangChain]
        LangGraph[LangGraph]
        Ollama[Ollama]
        Remote[Remote API]
    end

    subgraph Data["Data & Cache"]
        ClickHouse[ClickHouse]
        ChromaDB[ChromaDB]
        Redis[Redis]
    end

    subgraph External["External"]
        ai_toolkit[ai-toolkit]
    end

    subgraph Infra["Infrastructure"]
        Docker[Docker]
        DockerCompose[Docker Compose]
    end

    Framework --> Agent
    Agent --> Data
    Agent --> External
    Infra --> Framework
```

---

## 11. Directory Structure

```
IC-RAG-Agent/
├── src/
│   ├── uds/           # UDS Agent (BI for Amazon data)
│   ├── rag/           # RAG pipeline (intent + retrieval)
│   ├── agent/         # ReAct agent core
│   ├── sp_api/        # SP-API Agent

│   ├── tools/         # Shared utilities
│   └── draft/         # Prototypes
├── docs/
│   ├── guides/        # User, Developer, API, Deployment, Operations
│   ├── archive/       # Historical docs
│   └── FRAMEWORK.md   # This file
├── tests/
├── scripts/
├── docker/
├── monitoring/
├── external/
│   ├── ai-toolkit/    # BaseTool, ToolExecutor
│   ├── IC-Agent-Skill/
│   └── llama.cpp/
└── data/
```

---

## 12. Key Integration Points

| From | To | Purpose |
|------|-----|---------|
| UDS API | UDS Agent | Query processing |
| UDS Agent | ReAct Agent | Tool orchestration |
| UDS Agent | Intent Classifier | Domain classification |
| UDS Agent | Task Planner | Subtask decomposition |
| UDS Tools | UDS Client | ClickHouse queries |
| UDS Client | Redis | Query caching |
| RAG Pipeline | ChromaDB | Document retrieval |
| SP-API Agent | Long-term Memory | Uses RAG embeddings |
| All Agents | ai-toolkit | BaseTool, ToolExecutor |

---

## Related Documentation

- [PROJECT.md](PROJECT.md) – Project summary, metrics
- [OPERATIONS.md](OPERATIONS.md) – Operations manual
- [guides/UDS_DEVELOPER_GUIDE.md](guides/UDS_DEVELOPER_GUIDE.md) – Developer guide
- [archive/ARCHITECTURE_DECISIONS.md](archive/ARCHITECTURE_DECISIONS.md) – ADRs
