# SMT Ticketing Agentic Framework - Complete Implementation Guide

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PostgreSQL    │───▶│  Vector Database │───▶│   LLM Agent     │
│   (Tickets)     │    │   (ChromaDB)     │    │   (OpenAI)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Choreographer   │───▶│  Self-Healing   │
│  (New Ticket)   │    │    (Agent)       │    │    Service      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Technology Stack

### Core Components
- **Vector Database**: ChromaDB (open-source, Python-native)
- **LLM Integration**: OpenAI API with LangChain
- **Agent Framework**: CrewAI or LangGraph (open-source alternatives to expensive platforms)
- **Embeddings**: OpenAI text-embedding-3-small
- **Web Service**: FastAPI for script execution endpoints
- **Database**: PostgreSQL with psycopg2

### Alternative Paid Platforms (for comparison)
- **Pinecone** (vector DB) - $70/month for production
- **LangSmith** (LLM ops) - $39/month
- **Weights & Biases** (ML ops) - $50/month

## Implementation Steps

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv smt_agent_env
source smt_agent_env/bin/activate  # Windows: smt_agent_env\Scripts\activate

# Install dependencies
pip install chromadb langchain openai crewai fastapi uvicorn psycopg2-binary python-dotenv pydantic
```

### Step 2: Project Structure

```
smt_agentic_framework/
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── __init__.py
│   ├── postgres_connector.py
│   └── vector_store.py
├── agents/
│   ├── __init__.py
│   ├── choreographer.py
│   ├── similarity_agent.py
│   └── healing_agent.py
├── services/
│   ├── __init__.py
│   ├── script_executor.py
│   └── api_service.py
├── utils/
│   ├── __init__.py
│   └── embeddings.py
├── main.py
├── requirements.txt
└── .env
```

### Step 3: Configuration Setup

**`.env` file:**
```
OPENAI_API_KEY=your_openai_api_key
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=smt_tickets
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
CHROMA_PERSIST_DIR=./chroma_db
```

### Step 4: Core Implementation

#### 4.1 PostgreSQL to Vector Database Pipeline

```python
# data/postgres_connector.py
import psycopg2
import pandas as pd
from typing import List, Dict
import os
from dotenv import load_dotenv

class PostgresConnector:
    def __init__(self):
        load_dotenv()
        self.connection_params = {
            'host': os.getenv('POSTGRES_HOST'),
            'port': os.getenv('POSTGRES_PORT'),
            'database': os.getenv('POSTGRES_DB'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
    
    def extract_tickets(self) -> List[Dict]:
        """Extract closed tickets with solutions from PostgreSQL"""
        query = """
        SELECT 
            t.ticket_id,
            m.machine_name,
            o.name as operator_name,
            e.name as engineer_name,
            e.specialization,
            t.issue_description,
            t.status,
            t.priority,
            t.created_at,
            t.updated_at,
            ts.solution_text,
            ts.resolved_at
        FROM genschema.tickets t
        LEFT JOIN genschema.ticket_solutions ts ON t.ticket_id = ts.ticket_id
        LEFT JOIN genschema.machines m ON t.machine_id = m.machine_id
        LEFT JOIN genschema.operators o ON t.operator_id = o.operator_id
        LEFT JOIN genschema.engineers e ON ts.engineer_id = e.engineer_id
        WHERE t.status = 'Closed'
        AND ts.solution_text IS NOT NULL
        ORDER BY t.created_at DESC
        """
        
        with psycopg2.connect(**self.connection_params) as conn:
            df = pd.read_sql(query, conn)
            return df.to_dict('records')
```

#### 4.2 Vector Database Implementation

```python
# data/vector_store.py
import chromadb
from chromadb.config import Settings
import openai
from typing import List, Dict
import os

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
        )
        self.collection = self.client.get_or_create_collection(
            name="smt_tickets",
            metadata={"hnsw:space": "cosine"}
        )
        openai.api_key = os.getenv('OPENAI_API_KEY')
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        response = openai.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [data.embedding for data in response.data]
    
    def populate_from_tickets(self, tickets: List[Dict]):
        """Populate vector database from ticket data"""
        documents = []
        metadatas = []
        ids = []
        
        for ticket in tickets:
            # Create comprehensive document for embedding using actual schema
            document = f"""Issue: {ticket['issue_description']}
Machine: {ticket['machine_name']}
Priority: {ticket['priority']}
Solution: {ticket['solution_text']}
Engineer: {ticket['engineer_name']} ({ticket['specialization']})
Operator: {ticket['operator_name']}"""
            
            documents.append(document)
            
            metadatas.append({
                'ticket_id': str(ticket['ticket_id']),
                'machine_name': ticket['machine_name'],
                'priority': ticket['priority'],
                'engineer_specialization': ticket.get('specialization', ''),
                'solution_text': ticket['solution_text'],
                'resolved_at': str(ticket['resolved_at']),
                'issue_type': self._extract_issue_type(ticket['issue_description'])
            })
            
            ids.append(f"ticket_{ticket['ticket_id']}")
        
        # Generate embeddings
        embeddings = self.get_embeddings(documents)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} tickets to vector database")
    
    def _extract_issue_type(self, description: str) -> str:
        """Extract issue type from description"""
        description_lower = description.lower()
        issue_types = {
            'nozzle': 'nozzle pickup error',
            'spi': 'spi height deviation',
            'aoi': 'aoi false positive',
            'reflow': 'reflow oven temperature',
            'feeder': 'feeder jammed',
            'solder': 'solder bridging',
            'component': 'component misalignment',
            'tombstone': 'component tombstoning',
            'paste': 'solder paste misprint',
            'pcb': 'pcb warping'
        }
        
        for key, issue_type in issue_types.items():
            if key in description_lower:
                return issue_type
        return 'unknown'
    
    def search_similar_tickets(self, query: str, n_results: int = 3) -> Dict:
        """Search for similar tickets"""
        query_embedding = self.get_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results
```

#### 4.3 Agent Implementation using CrewAI

```python
# agents/choreographer.py
from crewai import Agent, Task, Crew
from langchain.llms import OpenAI
from data.vector_store import VectorStore
from typing import Dict, List
import json

class ChoreographerAgent:
    def __init__(self):
        self.llm = OpenAI(temperature=0.1)
        self.vector_store = VectorStore()
        
        # Define agents
        self.similarity_agent = Agent(
            role='Ticket Similarity Analyst',
            goal='Find similar resolved tickets and provide recommendations',
            backstory='Expert in analyzing ticket patterns and finding relevant solutions',
            llm=self.llm,
            verbose=True
        )
        
        self.decision_agent = Agent(
            role='Decision Maker',
            goal='Decide if automatic healing should be applied',
            backstory='Expert in assessing risk and determining automation feasibility',
            llm=self.llm,
            verbose=True
        )
    
    def process_new_ticket(self, ticket_description: str) -> Dict:
        """Process new ticket through the agent workflow"""
        
        # Search for similar tickets
        similar_tickets = self.vector_store.search_similar_tickets(ticket_description)
        
        # Create similarity analysis task
        similarity_task = Task(
            description=f"""
            Analyze the new SMT ticket: "{ticket_description}"
            
            Based on these similar resolved tickets from our database:
            {json.dumps(similar_tickets, indent=2)}
            
            Provide analysis for:
            1. Similarity assessment (0-100% match) - focus on issue type and machine compatibility
            2. Recommended solution approach based on successful past resolutions
            3. Confidence level in the recommendation (0-100%)
            4. Available resolution steps from similar cases
            5. Required engineer specialization
            
            Consider SMT-specific factors:
            - Machine type compatibility (Pick and Place, Reflow Oven, AOI, SPI)
            - Issue patterns (nozzle errors, height deviations, false positives, etc.)
            - Priority level alignment
            - Solution success history
            """,
            agent=self.similarity_agent,
            expected_output="JSON format with similarity_score, recommendation, confidence, available_steps, and required_specialization fields"
        )
        
        # Create decision task
        decision_task = Task(
            description=f"""
            Based on the similarity analysis, decide if automatic healing is recommended for this SMT issue.
            
            Consider SMT-specific criteria:
            - Similarity score threshold (>85% for auto-healing on SMT equipment)
            - Confidence level (>80% for auto-healing)
            - Risk assessment for production line impact
            - Solution step availability and clarity
            - Engineer specialization requirements
            
            SMT Auto-healing guidelines:
            - LOW RISK: Nozzle replacements, SPI recalibrations, AOI threshold adjustments
            - MEDIUM RISK: Feeder adjustments, temperature profile tweaks
            - HIGH RISK: Major reflow profile changes, mechanical alignments
            
            Provide recommendation: AUTO_HEAL, MANUAL_REVIEW, or ESCALATE_TO_ENGINEER
            """,
            agent=self.decision_agent,
            expected_output="JSON format with decision, reasoning, risk_level, and recommended_engineer_type fields"
        )
        
        # Execute crew
        crew = Crew(
            agents=[self.similarity_agent, self.decision_agent],
            tasks=[similarity_task, decision_task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            'similar_tickets': similar_tickets,
            'analysis': result,
            'ticket_description': ticket_description
        }
```

#### 4.4 Self-Healing Service

```python
# services/script_executor.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import subprocess
import json
from typing import Dict, Any

app = FastAPI(title="SMT Self-Healing Service")

class HealingRequest(BaseModel):
    ticket_id: str
    issue_description: str
    machine_name: str
    recommended_steps: str
    confidence_score: float
    engineer_specialization: str

@app.post("/execute-healing")
async def execute_healing_endpoint(request: HealingRequest):
    """Execute SMT healing procedures"""
    try:
        # Log the healing attempt
        print(f"SMT Auto-Healing Request:")
        print(f"Ticket ID: {request.ticket_id}")
        print(f"Machine: {request.machine_name}")
        print(f"Issue: {request.issue_description}")
        print(f"Recommended Steps: {request.recommended_steps}")
        print(f"Confidence: {request.confidence_score}%")
        print(f"Required Specialization: {request.engineer_specialization}")
        
        # Simulate SMT healing execution
        healing_steps = request.recommended_steps.split('\n')
        executed_steps = []
        
        for i, step in enumerate(healing_steps, 1):
            if step.strip():
                print(f"Executing Step {i}: {step.strip()}")
                executed_steps.append(f"✓ Step {i}: {step.strip()}")
                # Simulate processing time
                import time
                time.sleep(0.5)
        
        return {
            "status": "success",
            "ticket_id": request.ticket_id,
            "machine": request.machine_name,
            "executed_steps": executed_steps,
            "healing_duration": f"{len(executed_steps) * 0.5} seconds",
            "message": f"SMT auto-healing completed for {request.machine_name}",
            "next_action": "Monitor system for 10 minutes and verify resolution"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Healing failed: {str(e)}",
            "recommendation": "Escalate to engineer"
        }

class HealingAgent:
    def __init__(self):
        self.healing_service_url = "http://localhost:8001"
    
    def trigger_healing(self, healing_request: HealingRequest) -> Dict[str, Any]:
        """Trigger healing through web service"""
        try:
            response = requests.post(
                f"{self.healing_service_url}/execute-healing",
                json=healing_request.dict(),
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Service unavailable: {str(e)}
