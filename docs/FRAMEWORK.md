# IC-RAG-Agent System Framework

**Version:** 2.3.0  
**Last Updated:** 2026-03-08

This document describes the system framework for the IC-RAG-Agent project using Mermaid diagrams.

**How to view diagrams:** Open this file in Markdown preview mode (Mermaid rendering enabled).

---

## 1. System Overview

IC-RAG-Agent is an **Intent Classification + Retrieval-Augmented Generation** system with a **Unified Gateway** routing queries to five backend workflows:

- **Gateway** – Single entry point with clarification (required), Route LLM (rewriting + task classification), and Dispatcher (supervisor agent; executes worker agents in parallel)
- **UDS Agent** – Business Intelligence for Amazon seller data (ClickHouse + ReAct)
- **RAG Pipeline** – Document retrieval and hybrid generation with four parallel intent methods
- **SP-API Agent** – Seller Operations via Amazon SP-API (ReAct + LangGraph workflow)
- **Client** – Unified Gradio Chat UI calling the gateway

```mermaid
%%{init: {'themeVariables': {'fontSize': '11px'}, 'flowchart': {'curve': 'linear'}}}%%
flowchart TB
    subgraph Client["Client Layer"]
        UI[Unified Chat UI<br/>Gradio]
    end

    subgraph Gateway["Unified Gateway"]
        API[FastAPI]
        RouteLLM[Route LLM: Clarification, Rewriting, Intent Classification]
        Dispatcher[Dispatcher: Task Planning, Invoke, Merge]
    end

    subgraph Workers["Worker Agents"]
        RAG[RAG Pipeline]
        UDS[UDS Agent]
        SP[SP-API Agent]
    end

    subgraph Data["Data Layer"]
        Chroma[(ChromaDB)]
        Redis[(Redis)]
        CH[(ClickHouse)]
        SPAPI_EXT[Amazon SP-API]
    end

    subgraph Observability["Observability"]
        Monitor[Monitor System<br/>Grafana]
        Log[Log System<br/>Query Logs]
    end

    UI -->|POST| API
    API --> RouteLLM
    RouteLLM --> Dispatcher
    Dispatcher --> RAG
    Dispatcher --> UDS
    Dispatcher --> SP

    RAG --> Chroma
    RAG --> Redis
    RAG -->|logs| CH
    UDS --> CH
    UDS --> Redis
    SP --> SPAPI_EXT
    SP --> Redis
    SP -->|logs| CH

    API -->|logs| Log
    Log -->|write| CH
    Monitor -->|read| CH
```

> **Note:** Target design: Task planning moves to Dispatcher. Route LLM outputs intents only; Dispatcher maps intents to workflows and plans execution.

---

## 1.1 Five Workflows

| # | Workflow | Gateway Route | Backend | Port | Data Source | Status |
|---|----------|---------------|---------|------|-------------|--------|
| 1 | General Knowledge | `general` | RAG (general mode) | 8002 | Remote LLM (DeepSeek / Ollama) | ✅ Ready |
| 2 | Amazon Document | `amazon_docs` | RAG (documents mode) | 8002 | ChromaDB retrieval | ✅ Ready |
| 3 | Enterprise/IC Document | `ic_docs` | RAG (documents mode) | 8002 | ChromaDB (not populated) | ⚠️ Placeholder |
| 4 | SP-API Agent | `sp_api` | SP-API Agent | 8003 | Amazon Seller API | ✅ Ready |
| 5 | UDS Agent | `uds` | UDS Agent | 8001 | ClickHouse (40M+ rows) | ✅ Ready |

> **IC docs:** Not ready yet — Chroma not populated. Gateway returns a friendly message; set `IC_DOCS_ENABLED=true` once populated.

### 1.2 Gateway Grouping: Route LLM vs Dispatcher

The gateway is organized into two conceptual groups:

| Group | Responsibility | Modules | Description |
|-------|----------------|---------|-------------|
| **Route LLM** | Planning | `clarification.py`, `rewriters.py`, `router.py`, `route_llm.py` | Clarification (required), query rewriting, and task classification. Produces an execution plan (what to do). |
| **Dispatcher** | Execution | `api.py`, `services.py` | Supervisor agent; invokes worker agents (General RAG, Amazon docs RAG, SP-API Agent, UDS Agent) and executes tasks in parallel within groups, merges results. |

**Route LLM** outputs: rewritten query, execution plan (task_groups with workflow + query per task).

**Dispatcher** inputs: execution plan. Outputs: task_results, merged_answer, aggregated sources.

### 1.3 Role Analogy (Target Design)

| Role | Responsibility | Module |
|------|----------------|--------|
| **Decision Maker (Reason LLM)** | Clarify needs, identify intents | Route LLM |
| **Project Manager (Supervisor)** | Task planning, assignment, supervision, result aggregation | Dispatcher |
| **Worker** | Execute tasks, report results | RAG, SP-API, UDS |

**Proposed change:** Move Task planning from Route LLM to Dispatcher. Route LLM outputs intents only; Dispatcher performs intent → workflow mapping and task planning. See [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) for rationale and improvement suggestions.

### 1.4 Memory Strategy

| Layer | Store | Purpose |
|-------|-------|---------|
| **Short-term** | Redis | Session-scoped conversation history, multi-turn context. TTL-based expiration (e.g., 24h). Fast access for real-time follow-up questions. |
| **Long-term** | ClickHouse | Query logs, audit trails, analytics. Historical retention for dashboards, evaluation, debugging. |

**Usage:** RAG Pipeline, UDS Agent, and SP-API Agent use Redis for short-term memory (session history, cache). Query logs and long-term analytics are stored in ClickHouse.

---

## 2. Architecture Layers

```mermaid
flowchart TB
    subgraph L0["Client Layer"]
        Gradio[Gradio Chat UI]
        APIClient[GatewayClient<br/>api_client.py]
    end

    subgraph L1["Gateway Layer"]
        GatewayAPI[FastAPI Gateway]
        subgraph RL["Route LLM (Planning)"]
            Clarify[Clarification<br/>check_ambiguity]
            Rewriters[Query Rewriters<br/>Ollama / DeepSeek]
            Planner[Planner + Correction]
            RouteLLMMod[Route LLM / Heuristic]
        end
        subgraph Disp["Dispatcher (Execution)"]
            Orch[Orchestrator<br/>api.py]
            Services[Service Dispatch<br/>services.py]
        end
        LogUtils[Logging Utils]
    end

    subgraph L2["API Layer"]
        UDS_API[UDS FastAPI :8001]
        RAG_API[RAG FastAPI :8002]
        SP_API[SP-API FastAPI :8003]
    end

    subgraph L3["Agent Layer"]
        ReAct[ReAct Agent Core]
        UDS_Intent[UDS Intent Classifier]
        RAG_Intent[RAG Intent Methods ×4]
        Planner[Task Planner]
    end

    subgraph L4["Tool Layer"]
        UDS_Tools[UDS Tools<br/>Schema 4 + Query 4 + Analysis 5 + Viz 3]
        SP_Tools[SP-API Tools<br/>10 tools]
    end

    subgraph L5["Data Layer"]
        ClickHouse[(ClickHouse)]
        ChromaDB[(ChromaDB)]
        Redis[(Redis Cache)]
    end

    subgraph L6["LLM Layer"]
        Ollama[Ollama<br/>Qwen3, qwen2.5]
        DeepSeek[DeepSeek API]
        HF[HuggingFace<br/>MiniLM, DistilBERT]
    end

    L0 --> L1 --> L2 --> L3 --> L4 --> L5
    L3 --> L6
```

---

## 3. Module Structure

```mermaid
flowchart LR
    subgraph src["src/"]
        subgraph gateway["gateway/"]
            subgraph route_llm["Route LLM"]
                gw_clarify[clarification.py]
                gw_router[router.py]
                gw_rewriters[rewriters.py]
                gw_route_llm[route_llm.py]
            end
            subgraph dispatcher["Dispatcher"]
                gw_api[api.py]
                gw_services[services.py]
            end
            gw_schemas[schemas.py]
            gw_log[logging_utils.py]
        end

        subgraph client["client/"]
            cl_api[api_client.py]
            cl_ui[gradio_ui.py]
        end

        subgraph uds["uds/"]
            u_api[api.py]
            u_agent[uds_agent.py]
            u_client[uds_client.py]
            u_intent[intent_classifier.py]
            u_planner[task_planner.py]
            u_formatter[result_formatter.py]
            u_cache[cache.py]
            u_error[error_handler.py]
            u_schemas[schemas.py]
            u_tools[tools/]
            u_maint[maintenance/]
            u_data[data/]
        end

        subgraph rag["rag/"]
            r_api[rag_api.py]
            r_pipeline[query_pipeline.py]
            r_intent_m[intent_methods.py]
            r_aggregator[intent_aggregator.py]
            r_classifier[intent_classifier.py]
            r_rewrite[query_rewriting.py]
            r_chroma[chroma_loaders.py]
            r_embed[embeddings.py]
            r_ingest[ingest_pipeline.py]
            r_eval[evaluation/]
        end

        subgraph agent_mod["agent/"]
            a_react[react_agent.py]
            a_models[models.py]
            a_logger[agent_logger.py]
            a_except[exceptions.py]
            a_tools[tools/]
        end

        subgraph sp_api["sp_api/"]
            s_api[fast_api.py]
            s_agent[sp_api_agent.py]
            s_client[sp_api_client.py]
            s_workflow[workflow.py<br/>LangGraph]
            s_ltm[long_term_memory.py<br/>ChromaDB]
            s_stm[short_term_memory.py<br/>Redis]
            s_schemas[schemas.py]
            s_tools[tools/ ×10]
        end
    end

    subgraph external["external/"]
        ai_toolkit[ai-toolkit<br/>BaseTool, ToolExecutor]
        ic_skill[IC-Agent-Skill]
        llama[llama.cpp]
    end

    u_agent --> a_react
    s_agent --> a_react
    a_react --> ai_toolkit
    cl_ui --> cl_api
    cl_api --> gw_api
    gw_api --> gw_router
    gw_api --> gw_services
    gw_router --> gw_route_llm
    gw_router --> gw_rewriters
```

---

## 4. Gateway Flow

```mermaid
sequenceDiagram
    participant User
    participant ChatUI as Gradio Chat UI
    participant GW as Gateway API
    participant Clarify as Clarification
    participant RouteLLM as Route LLM
    participant Dispatcher as Dispatcher
    participant Backend as Backend Services

    User->>ChatUI: Type question
    ChatUI->>GW: POST /api/v1/query<br/>{query, workflow, rewrite_enable, session_id}
    GW->>Clarify: raw_query

    alt GATEWAY_CLARIFICATION_ENABLED and query ambiguous
        Clarify->>Clarify: check_ambiguity (LLM)
        Clarify-->>GW: needs_clarification, clarification_question
        GW-->>ChatUI: QueryResponse (clarification_required, pending_query)
        ChatUI-->>User: Display clarification question
        User->>ChatUI: Follow-up answer
        ChatUI->>GW: POST (merged: pending_query + follow-up)
    end

    GW->>RouteLLM: raw_query (or merged)

    Note over RouteLLM: Rewriting + Task Classification
    RouteLLM->>RouteLLM: Normalize, LLM rewrite, plan/correct
    RouteLLM-->>GW: execution_plan (rewritten_query, task_groups)

    GW->>Dispatcher: execution_plan

    Note over Dispatcher: Supervise + Invoke
    Dispatcher->>Backend: Execute tasks (parallel/sequential)
    Backend-->>Dispatcher: task_results
    Dispatcher->>Dispatcher: Merge answers
    Dispatcher-->>GW: merged_answer, task_results

    GW-->>ChatUI: QueryResponse (answer, plan, task_results)
    ChatUI-->>User: Display answer

    Note over GW: Clarification (before rewrite). Route LLM = planning. Dispatcher = execution.
```

---

## 5. Routing Logic

```mermaid
flowchart TD
    Q[User Query] --> CLARIFY{GATEWAY_CLARIFICATION_ENABLED?}
    CLARIFY -->|No| WF
    CLARIFY -->|Yes| AMBIG{check_ambiguity<br/>LLM}
    AMBIG -->|needs_clarification| ASK[Return clarification_question<br/>pending_query; user follows up]
    AMBIG -->|clear| WF
    
    WF[workflow field?] -->|explicit: general/uds/sp_api/...| MANUAL[Manual Override<br/>confidence=1.0]
    WF -->|auto| AUTO{Planner enabled?}
    
    AUTO -->|Yes| PLAN[Parse planner JSON<br/>or heuristic multi-task]
    AUTO -->|No| ROUTE{Route LLM<br/>enabled?}
    
    PLAN --> CORRECT[Plan Correction<br/>heuristic override ≥0.9]
    CORRECT --> EXEC[Execute plan]
    
    ROUTE -->|No| HEUR[Keyword Heuristic]
    ROUTE -->|Yes| LLM[Route LLM<br/>Ollama / DeepSeek]
    
    LLM --> CONF{confidence ≥<br/>threshold?}
    CONF -->|Yes| USE_LLM[Use LLM result]
    CONF -->|No| HEUR
    
    HEUR --> PRIORITY[Priority-ordered keyword blocks]
    PRIORITY -->|sp_api 0.92| SP_H[order status, inventory health, buy box, etc.]
    PRIORITY -->|uds 0.92| UD_H[top 5, products by, by refund, etc.]
    PRIORITY -->|uds 0.85| UD_M[table, schema, trend, total, etc.]
    PRIORITY -->|amazon_docs 0.9| AM[policy, fees, guidelines, etc.]
    PRIORITY -->|sp_api 0.85| SP_M[sp-api, fba, shipment, catalog]
    PRIORITY -->|no match| W1[general 0.7]
```

### 5.1 Clarification (Question User)

Clarification is required by default. The gateway runs a clarification check **before** rewriting (set `GATEWAY_CLARIFICATION_ENABLED=false` to disable). The LLM (`check_ambiguity`) detects ambiguous queries (e.g. "Show me the fees" without specifying fee type or period) and returns a clarification question. The client stores `pending_query` and merges the user's follow-up with it on the next turn. If the query is clear, the flow proceeds to rewriting and execution.

**Example:** User asks "Show me the fees" → Gateway returns "Which type of fees do you mean? FBA, storage, or referral?" → User replies "FBA fees for last month" → Client sends merged query "Show me the fees FBA fees for last month" → Gateway proceeds to rewrite and route.

**Why clarification before rewriting:** Ambiguity is about missing information (which fees? what period?). The rewriter cannot invent information; it would guess and introduce bias. Clarifying first avoids wasted rewrite calls and ensures correct routing. See [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md).

### 5.2 Route LLM Steps (Current)

1. **Clarification** (required) – Detect ambiguous queries; ask user.
2. **Normalize** – Trim and collapse whitespace.
3. **Rewrite** – LLM rewrites or classifies intents.
4. **Build execution plan** – Parse planner output; route intents to workflows, create task_groups.
5. **Plan correction** – Heuristic override for misclassifications.
6. **Expand merged tasks** – Split tasks with multiple sub-questions.

### 5.3 Query Rewriting and Routing Rules

**File locations:**

| Component | File | Description |
|-----------|------|--------------|
| Clarification | `src/gateway/clarification.py` | `check_ambiguity()` – LLM detects ambiguous queries before rewrite; asks user for clarification. Required by default; set `GATEWAY_CLARIFICATION_ENABLED=false` to disable. |
| Rewrite prompts | `src/gateway/rewriters.py` | `REWRITE_PROMPT`, `REWRITE_PLANNER_PROMPT`, `INTENT_CLASSIFICATION_PROMPT` |
| Intent classification | `src/gateway/rewriters.py` | `rewrite_intents_only()` – Phase 1 of two-phase flow |
| Heuristic split | `src/gateway/router.py` | `_split_multi_intent_clauses()` – fallback when LLM fails |
| Heuristic routing | `src/gateway/router.py` | `_route_workflow_heuristic()` |
| Plan correction | `src/gateway/router.py` | `_correct_plan_workflows()` |
| Route LLM prompt | `src/gateway/route_llm.py` | `ROUTE_LLM_SYSTEM_PROMPT` |

**Planner routing policy (rewriters.py):**

- **amazon_docs:** Amazon business rules, policies, requirements, fee definitions
- **uds:** Analytical/historical questions; prefer UDS when data is in warehouse snapshots
- **sp_api:** Real-time/current-state data only; SP-API has rate limits
- **sp_api must not** handle policy/business-rule explanation

**Heuristic keywords (router.py, priority order):**

| Workflow | Confidence | Keywords (examples) |
|----------|------------|---------------------|
| sp_api | 0.92 | order status, check order, inventory placement, inventory health, buy box status, check if asin, verify my seller |
| uds | 0.92 | top 5, top 10, products by, by refund, by product, refund count |
| uds | 0.85 | which table, schema, dataset, clickhouse, last month, trend, total, average, compare, breakdown, historical |
| amazon_docs | 0.9 | policy, requirements, fees, fee structure, guidelines, what does amazon |
| sp_api | 0.85 | sp-api, fba, shipment, catalog, seller api |
| general | 0.7 | fallback when no keyword match |

### 5.4 Multi-Task Execution Flow (Two-Phase Intent Split)

When `GATEWAY_REWRITE_PLANNER_ENABLED=true`, the gateway uses a two-phase flow to avoid LLM merging multiple sub-questions into one task:

1. **Phase 1 – Intent classification:** `rewrite_intents_only()` calls the LLM to list distinct sub-questions. Output: `{"intents": ["...", "..."]}`. On success, tasks are built from intents. On failure, heuristic split is used.

2. **Phase 2 – Task building:** For each intent, `_route_workflow_heuristic()` assigns a workflow. One task per intent.

**Heuristic split fallback:** When Phase 1 fails, `_split_multi_intent_clauses()` splits the query by question-starter patterns (`get order`, `which`, `show me`, `what is`, etc.). Example: `"what is FBA get order 123 which table show me trend"` → 4 clauses → 4 tasks.

```mermaid
flowchart TB
    Q[User Query] --> PH1{Phase 1:\nIntent Classification}
    PH1 -->|LLM success| INT[Parse intents JSON]
    PH1 -->|LLM fail| HEUR[Heuristic split\n_split_multi_intent_clauses]
    INT --> ROUTE[Route each intent\n_route_workflow_heuristic]
    HEUR --> ROUTE
    ROUTE --> PLAN[Planner\nTask groups]
    PLAN --> CORR[Plan Correction\nHeuristic override]
    CORR --> SPLIT{Split into Tasks}
    
    SPLIT -->|general| G[General RAG\nRemote LLM]
    SPLIT -->|amazon_docs| A[Amazon Docs RAG\nChromaDB]
    SPLIT -->|ic_docs| IC[IC Docs RAG\nChromaDB]
    SPLIT -->|sp_api| SP[SP-API Agent\nAmazon API]
    SPLIT -->|uds| U[UDS Agent\nClickHouse]

    G -->|answer| M[Merge Answers\nconcat / compare / synthesize]
    A -->|answer| M
    IC -->|answer| M
    SP -->|answer| M
    U -->|answer| M

    M --> R[Final Response to User]

    style PH1 fill:#f9f,stroke:#333
    style PLAN fill:#f9f,stroke:#333
    style CORR fill:#ff9,stroke:#333
    style M fill:#9ff,stroke:#333
```

---

## 6. UDS Agent Flow

```mermaid
flowchart TB
    subgraph Input
        Q[User Query]
    end

    subgraph UDS["UDS Agent Pipeline"]
        IC[Intent Classifier<br/>6 Domains]
        TP[Task Planner<br/>Subtask Decomposition]
        RA[ReAct Agent Loop]
        RF[Result Formatter]
        EH[Error Handler<br/>Circuit Breaker + Retry]
    end

    subgraph Tools["16 Tools"]
        Schema[Schema Tools ×4<br/>list_tables, describe_table<br/>get_relationships, search_columns]
        Query[Query Tools ×4<br/>generate_sql, execute_query<br/>validate_query, explain_query]
        Analysis[Analysis Tools ×5<br/>sales_trend, inventory<br/>product_perf, financial, compare]
        Viz[Visualization Tools ×3<br/>chart, dashboard, export]
    end

    subgraph Data
        CH[(ClickHouse<br/>40M+ rows)]
        Cache[(Redis<br/>Query + Schema Cache)]
    end

    Q --> IC --> TP --> RA --> Tools
    Tools --> CH
    RA --> RF --> Out[Response]
    IC -.-> Cache
    Tools -.-> Cache
    RA -.-> EH
```

---

## 7. RAG Pipeline Flow

```mermaid
flowchart TB
    subgraph Input
        Query[User Query]
    end

    subgraph Preprocess
        QR[Query Rewriting<br/>query_rewriting.py]
    end

    subgraph Parallel["Four Parallel Intent Methods"]
        M1[Documents Method<br/>Chroma L2 distance]
        M2[Keywords Method<br/>Domain phrase match]
        M3[FAQ Method<br/>Vector similarity]
        M4[LLM Method<br/>Zero-shot NLI<br/>DistilBERT]
    end

    subgraph Aggregate
        AGG[Aggregate Yes/No<br/>intent_aggregator.py]
    end

    subgraph Mode["Answer Mode Decision"]
        GEN["General Mode<br/>All No -> remote LLM"]
        DOC["Documents Mode<br/>One or more Yes -> local LLM plus context"]
        HYB["Hybrid Mode<br/>One or more Yes -> both LLMs"]
    end

    subgraph Generation
        Local[Local LLM<br/>Ollama Qwen3]
        Remote[Remote LLM<br/>DeepSeek API]
    end

    subgraph Evaluation["RAG Evaluation"]
        Recall["Recall at 5"]
        Precision["Precision at 5"]
        Faith["Faithfulness"]
        Relevance["Relevance"]
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

## 8. SP-API Agent Flow

```mermaid
flowchart TB
    subgraph Input
        Q[User Query]
    end

    subgraph SPAPI["SP-API Agent"]
        Intent[Intent Classify<br/>query / action / report]
        Select[Tool Selection<br/>by intent]
        React[ReAct Loop<br/>Thought→Action→Observation]
        Format[Format Response]
    end

    subgraph Workflow["LangGraph Workflow"]
        N1[classify_intent]
        N2[select_tools]
        N3[execute_react_loop]
        N4[format_response]
        N5[store_memory]
        N1 --> N2 --> N3 --> N4 --> N5
    end

    subgraph Tools["10 Tools"]
        T1[ProductCatalogTool]
        T2[InventoryTool]
        T3[ListOrdersTool]
        T4[OrderDetailsTool]
        T5[ListShipmentsTool]
        T6[CreateShipmentTool]
        T7[FBAFeeTool]
        T8[FBAEligibilityTool]
        T9[FinancialsTool]
        T10[ReportRequestTool]
    end

    subgraph Memory["Memory Layer"]
        STM[Short-term Memory<br/>Redis / ConversationMemory]
        LTM[Long-term Memory<br/>ChromaDB Embeddings]
    end

    Q --> Intent --> Select --> React --> Format
    React --> Tools
    Tools -->|Amazon API| SPAPI_EXT[Amazon SP-API]
    React -.-> STM
    React -.-> LTM
```

---

## 9. ReAct Agent Loop

```mermaid
flowchart LR
    subgraph ReAct["ReAct Loop (max_iterations)"]
        T["Thought<br/>LLM reasoning"]
        A["Action<br/>Tool selection + args"]
        O["Observation<br/>Tool result"]
    end

    T --> A --> O --> T

    subgraph Infra["Agent Infrastructure"]
        Logger[AgentLogger<br/>Structured logging]
        State[AgentState<br/>history, iterations]
        Errors[MaxIterationsError<br/>ToolNotFoundError]
    end

    subgraph ToolExec["ai-toolkit"]
        TE[ToolExecutor]
        BT[BaseTool]
    end

    A --> TE --> BT
    BT --> O
    ReAct -.-> Infra
```

---

## 10. Data Flow

```mermaid
flowchart TB
    subgraph Inbound
        U[User Request]
    end

    subgraph Gateway
        GW[Gateway API :8000]
    end

    subgraph Processing
        RAG_SVC[RAG Pipeline :8002]
        UDS_SVC[UDS Agent :8001]
        SP_SVC[SP-API Agent :8003]
    end

    subgraph Storage
        CH[(ClickHouse<br/>40M+ rows<br/>amz_order, inventory, etc.)]
        Chroma[(ChromaDB<br/>Collections: documents,<br/>fqa_question, keyword)]
        Redis[(Redis<br/>Query cache, session,<br/>schema metadata)]
        LTM_Store[(ChromaDB<br/>SP-API long-term memory)]
    end

    subgraph Outbound
        R[Response]
    end

    U --> GW
    GW --> RAG_SVC --> Chroma
    GW --> UDS_SVC --> CH
    GW --> SP_SVC --> LTM_Store
    UDS_SVC -.-> Redis
    SP_SVC -.-> Redis
    RAG_SVC --> R
    UDS_SVC --> R
    SP_SVC --> R
```

---

## 11. Deployment Architecture

```mermaid
flowchart TB
    subgraph Cloud["Alibaba Cloud ECS"]
        subgraph Nginx_LB["Nginx :80"]
            LB[Reverse Proxy<br/>Rate Limiting<br/>Security Headers]
        end

        subgraph AppServices["Application Services"]
            GW[Gateway :8000]
            UDS[UDS Agent :8001]
            RAG[RAG Pipeline :8002]
            SP[SP-API Agent :8003]
            ChatUI[Chat UI :7862]
        end

        subgraph DataServices["Data Services (Docker)"]
            CH[ClickHouse :8123/:9000]
            R[Redis :6379]
            Chroma[ChromaDB :8000]
        end

        subgraph Monitoring["Monitoring"]
            Prom[Prometheus]
            Graf[Grafana]
            Alerts[Alert Rules]
        end
    end

    subgraph External["External Services"]
        Ollama[Ollama LLM<br/>qwen3, qwen2.5]
        DeepSeekAPI[DeepSeek API]
        AmazonAPI[Amazon SP-API]
    end

    subgraph CI["CI/CD"]
        GHA[GitHub Actions]
        ACR[Alibaba Container Registry]
    end

    LB -->|/api/v1/query| GW
    LB -->|/api/| UDS
    GW --> RAG & UDS & SP
    RAG --> Chroma
    UDS --> CH & R
    SP --> AmazonAPI
    GW & UDS & RAG & SP --> Ollama & DeepSeekAPI
    AppServices --> Prom --> Graf
    GHA --> ACR --> Cloud
```

---

## 12. Technology Stack

```mermaid
flowchart TB
    subgraph Framework["Web Framework"]
        FastAPI[FastAPI]
        Pydantic[Pydantic v2]
        Uvicorn[Uvicorn]
        Gradio[Gradio]
    end

    subgraph Agent["Agent & LLM"]
        LangChain[LangChain]
        LangGraph[LangGraph]
        Ollama[Ollama]
        DeepSeek[DeepSeek API]
        HF[HuggingFace<br/>Transformers]
    end

    subgraph Data["Data & Cache"]
        ClickHouse[ClickHouse]
        ChromaDB[ChromaDB]
        Redis[Redis]
    end

    subgraph Embedding["Embeddings"]
        MiniLM[all-MiniLM-L6-v2]
        DistilBERT[DistilBERT SST-2]
        Qwen3VL[Qwen3-VL-Embedding]
    end

    subgraph External["External Libraries"]
        ai_toolkit[ai-toolkit<br/>BaseTool, ToolExecutor]
        ic_skill[IC-Agent-Skill]
        llama_cpp[llama.cpp]
    end

    subgraph Infra["Infrastructure"]
        Docker[Docker + Compose]
        Nginx[Nginx]
        GitHub[GitHub Actions CI/CD]
        Prometheus[Prometheus + Grafana]
        Aliyun[Alibaba Cloud ECS]
    end

    Framework --> Agent --> Data
    Agent --> Embedding
    Agent --> External
    Infra --> Framework
```

---

## 13. Directory Structure

```
IC-RAG-Agent/
├── src/
│   ├── gateway/        # Unified gateway (routing, rewriting, dispatch)
│   │   ├── api.py            # FastAPI app, POST /api/v1/query
│   │   ├── clarification.py  # check_ambiguity – question user when query ambiguous
│   │   ├── router.py         # Route LLM + heuristic fallback
│   │   ├── rewriters.py      # Ollama / DeepSeek query rewriting
│   │   ├── route_llm.py      # LLM-based workflow classifier
│   │   ├── services.py       # Backend HTTP dispatch
│   │   ├── schemas.py        # QueryRequest / QueryResponse
│   │   └── logging_utils.py  # Structured routing log helpers
│   ├── client/         # Unified chat client
│   │   ├── api_client.py     # GatewayClient (HTTP + mock mode)
│   │   └── gradio_ui.py      # Gradio Chat UI
│   ├── uds/            # UDS Agent (BI for Amazon data)
│   │   ├── api.py            # FastAPI :8001 (sync + streaming)
│   │   ├── uds_agent.py      # UDSAgent extends ReActAgent
│   │   ├── uds_client.py     # ClickHouse client + retry
│   │   ├── intent_classifier.py
│   │   ├── task_planner.py
│   │   ├── result_formatter.py
│   │   ├── cache.py          # Redis cache wrapper
│   │   ├── error_handler.py  # Circuit breaker + retry + backoff
│   │   ├── schemas.py        # Pydantic request/response
│   │   ├── tools/            # 16 tools (schema, query, analysis, viz)
│   │   ├── maintenance/      # quality_checks, statistics
│   │   └── data/             # Business glossary, schema metadata
│   ├── rag/            # RAG Pipeline (intent + retrieval + generation)
│   │   ├── rag_api.py        # FastAPI :8002
│   │   ├── query_pipeline.py # RAGPipeline.build() + query()
│   │   ├── intent_methods.py # 4 parallel intent methods
│   │   ├── intent_aggregator.py
│   │   ├── intent_classifier.py  # Zero-shot NLI
│   │   ├── query_rewriting.py
│   │   ├── chroma_loaders.py
│   │   ├── embeddings.py
│   │   ├── ingest_pipeline.py
│   │   └── evaluation/       # RAGAS metrics, dataset loader, reports
│   ├── agent/          # ReAct agent core (shared by UDS + SP-API)
│   │   ├── react_agent.py    # ReActAgent class
│   │   ├── models.py         # Action, Observation, AgentState
│   │   ├── agent_logger.py   # Structured agent logging
│   │   ├── exceptions.py     # MaxIterationsError, ToolNotFoundError
│   │   └── tools/            # Stub tools (uds_stubs, sp_api_stubs)
│   ├── sp_api/         # SP-API Agent (Amazon seller operations)
│   │   ├── fast_api.py       # FastAPI :8003 (SSE streaming)
│   │   ├── sp_api_agent.py   # SellerOperationsAgent extends ReActAgent
│   │   ├── sp_api_client.py  # Amazon SP-API HTTP client
│   │   ├── workflow.py       # LangGraph state machine
│   │   ├── long_term_memory.py   # ChromaDB semantic memory
│   │   ├── short_term_memory.py  # Redis conversation memory
│   │   ├── schemas.py
│   │   └── tools/            # 10 tools (catalog, orders, inventory, etc.)
│   └── draft/          # Prototypes and experiments
├── scripts/
│   ├── run_gateway.py        # Gateway launcher
│   ├── run_unified_chat.py   # Chat UI launcher
│   ├── run_all_tests.py      # Test runner
│   ├── run_evaluation.py     # RAG evaluation runner
│   ├── load_to_chroma.py     # RAG ingest helper
│   ├── query_rag.py          # RAG query helper
│   └── run_sp_api_gradio.py  # SP-API UI launcher
├── docker/
│   ├── docker-compose.yml    # Redis + ClickHouse + ChromaDB + Gateway
│   ├── Dockerfile.gateway    # Gateway Docker image
│   ├── nginx.conf            # Reverse proxy + rate limiting
│   └── docker-compose.*.yml  # ECS, prod, UDS variants
├── monitoring/
│   ├── prometheus.yml        # Prometheus scrape config
│   ├── alert-rules.yml       # Alert rules
│   └── grafana-dashboards/   # Overview, performance, errors, infra
├── tests/                    # 60+ test files
│   ├── test_gateway_*.py     # Gateway API, router, rewriters, services
│   ├── test_client_*.py      # Client API, Gradio UI
│   ├── test_uds_*.py         # UDS agent, API, client, integration
│   ├── test_rag_*.py         # RAG API, intent, pipeline
│   ├── test_*_agent.py       # ReAct agent, SP-API agent
│   └── *.py                  # Cache, error handling, security, load, UAT
├── tools/                    # Dev/ops utilities
│   ├── benchmark_api.py
│   ├── optimize_queries.py
│   └── generate_*.py         # Quality reports, schema metadata
├── bin/                      # Shell scripts
│   ├── run_rag_api.sh
│   ├── uds_ops.sh
│   └── download_models_from_hf.sh
├── db/                       # Database DDL
│   └── uds/
│       ├── create_tables.sql
│       └── create_indexes.sql
├── specs/                    # API specifications
│   └── UDS_API_SPEC.yaml
├── .github/workflows/        # CI/CD
│   └── deploy.yml            # GitHub Actions → Alibaba Cloud
├── external/
│   ├── ai-toolkit/           # BaseTool, ToolExecutor
│   ├── IC-Agent-Skill/       # Skill definitions
│   └── llama.cpp/            # Local LLM inference
├── models/                   # Local model weights
│   ├── all-MiniLM-L6-v2/
│   ├── distilbert-base-uncased-finetuned-sst-2-english/
│   ├── Qwen3-1.7B/
│   └── Qwen3-VL-Embedding-*/
└── data/
    ├── documents/            # Source documents for RAG
    ├── chroma_db/            # ChromaDB persistent storage
    ├── vector_store/         # Additional vector stores
    └── intent_classification/ # Intent training data
```

---

## 14. Key Integration Points

| From | To | Method | Purpose |
|------|----|--------|---------|
| Chat UI | Gateway | HTTP POST | Unified entry point for all queries |
| Gateway (Route LLM) | Gateway (Dispatcher) | Function call | Pass execution_plan (planning -> execution) |
| Dispatcher | RAG API | HTTP POST `/query` | General, Amazon docs, IC docs workflows |
| Dispatcher | UDS API | HTTP POST `/api/v1/uds/query` | BI analytics queries |
| Dispatcher | SP-API | HTTP POST `/api/v1/seller/query` | Seller operations |
| Gateway Router | Route LLM | Function call | LLM-based workflow classification |
| Gateway Router | Heuristic | Function call | Keyword-based fallback routing |
| Gateway Rewriter | Ollama/DeepSeek | HTTP | Query rewriting before routing |
| UDS Agent | ReAct Agent | Inheritance | Tool orchestration loop |
| SP-API Agent | ReAct Agent | Inheritance | Tool orchestration loop |
| SP-API Agent | LangGraph | StateGraph | Workflow state machine |
| UDS Agent | ClickHouse | clickhouse-connect | SQL query execution |
| UDS Agent | Redis | redis-py | Query + schema caching |
| RAG Pipeline | ChromaDB | chromadb | Document retrieval (L2 distance) |
| RAG Pipeline | HuggingFace | transformers | Zero-shot NLI intent classification |
| SP-API Agent | ChromaDB | chromadb | Long-term semantic memory |
| SP-API Agent | Redis | redis-py | Short-term conversation memory |
| All Agents | ai-toolkit | import | BaseTool, ToolExecutor |
| Nginx | Gateway | proxy_pass | Reverse proxy + rate limiting |
| GitHub Actions | Alibaba Cloud | SSH deploy | CI/CD pipeline |

---

## 15. API Contract

### 15.0 Service Health Endpoints

| Service | Base Port | Health Endpoint |
|---------|-----------|-----------------|
| Gateway | 8000 | `/health` |
| UDS API | 8001 | `/health` |
| RAG API | 8002 | `/health` |
| SP-API API | 8003 | `/api/v1/health` |

### 15.1 Gateway Request

```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "What were my sales in October?",
  "workflow": "auto",
  "rewrite_enable": true,
  "rewrite_backend": "ollama",
  "route_backend": null,
  "session_id": "session-1234",
  "stream": false
}
```

### 15.2 Gateway Response

```json
{
  "answer": "Your total Amazon sales in October were $12,345.",
  "workflow": "uds",
  "routing_confidence": 0.96,
  "sources": [{"type": "table", "name": "amz_order"}],
  "request_id": "req-uuid",
  "error": null,
  "plan": {"plan_type": "hybrid", "task_groups": [...]},
  "task_results": [{"workflow": "uds", "status": "completed", "answer": "..."}],
  "merged_answer": "- [uds] Your total Amazon sales..."
}
```

When `GATEWAY_REWRITE_PLANNER_ENABLED=true`, the response includes `plan`, `task_results`, and `merged_answer` for multi-task execution.

---

## Related Documentation

- [PROJECT.md](PROJECT.md) – Project summary, metrics
- [OPERATIONS.md](OPERATIONS.md) – Operations manual
- [guides/UDS_DEVELOPER_GUIDE.md](guides/UDS_DEVELOPER_GUIDE.md) – Developer guide
- [guides/UDS_API_REFERENCE.md](guides/UDS_API_REFERENCE.md) – UDS API reference
- [guides/QUERY_REWRITING.md](guides/QUERY_REWRITING.md) – Query rewriting guide
- [archive/ARCHITECTURE_DECISIONS.md](archive/ARCHITECTURE_DECISIONS.md) – ADRs
- [archive/ANSWER_MODEL_IDENTITY.md](archive/ANSWER_MODEL_IDENTITY.md) – Answer model identity notes

