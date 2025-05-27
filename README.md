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


<img width="465" alt="image" src="https://github.com/user-attachments/assets/6bea9fff-1d50-4c96-9fd3-547b9732bcb3" />
