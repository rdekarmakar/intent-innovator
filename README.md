---
config:
  theme: neutral
---
sequenceDiagram
    participant App as Application
    participant API as FastAPI Layer
    participant LLM as AI/LLM Service
    
    App->>API: Send request with data
    API->>API: Validate data
    API->>LLM: Process with AI model
    LLM->>API: Return results
    API->>App: Deliver formatted response
