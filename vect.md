# SMT Ticket Resolution System - xAI Grok Edition

## Project Structure
```
smt_ticket_resolution_xai/
├── README.md
├── requirements.txt
├── .env.example
├── setup.py
├── setup_system.py
├── main.py
├── config/
│   ├── __init__.py
│   └── database.py
├── models/
│   ├── __init__.py
│   ├── database_manager.py
│   ├── vector_store.py
│   └── ticket_resolver.py
├── utils/
│   ├── __init__.py
│   └── preprocessing.py
├── data/
│   └── embeddings/
│       └── .gitkeep
├── tests/
│   ├── __init__.py
│   └── test_system.py
└── docs/
    └── API.md
```

## File Contents:

### README.md
```markdown
# SMT Ticket Resolution System - xAI Grok Edition

An intelligent ticket resolution system for Surface Mount Technology (SMT) manufacturing that uses vector similarity search and xAI's Grok-beta to recommend solutions based on historical ticket data.

## Features

- **Vector Similarity Search**: Uses Sentence-Transformers and FAISS for fast, accurate similarity matching
- **xAI Grok-Powered Recommendations**: Grok-beta generates detailed, actionable resolution steps
- **SMT Domain-Specific**: Tailored for PCB assembly and SMT manufacturing processes
- **Easy Integration**: Works with existing PostgreSQL databases
- **Interactive CLI**: Command-line interface for easy ticket resolution
- **No OpenAI Credits Required**: Uses free/local embeddings and xAI's competitive pricing

## Technology Stack

- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2) - Free & Local
- **LLM**: xAI Grok-beta
- **Database**: PostgreSQL
- **Framework**: LangChain
- **Language**: Python 3.8+

## Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd smt_ticket_resolution_xai
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your xAI API key
   ```

3. **Setup System**
   ```bash
   python setup_system.py
   ```

4. **Run Interactive Mode**
   ```bash
   python main.py --interactive
   ```

## xAI API Setup

1. Get your xAI API key from: https://console.x.ai/
2. Add it to your .env file as XAI_API_KEY
3. Grok-beta offers competitive pricing and excellent performance

## Use Cases

- **New Ticket Resolution**: Get instant recommendations for similar past issues
- **Knowledge Management**: Leverage historical solutions for faster problem-solving
- **Engineer Training**: Learn from expert solutions and best practices
- **Process Optimization**: Identify common issues and preventive measures

## SMT Domain Coverage

- Pick and Place machines (nozzle issues, component placement)
- Reflow ovens (temperature profiles, heating issues)
- AOI systems (false positives, calibration)
- SPI systems (height deviations, paste inspection)
- Component issues (tombstoning, misalignment, bridging)
- PCB issues (warping, substrate problems)

## Cost Benefits

- **Free Embeddings**: Uses local Sentence-Transformers models
- **Competitive LLM Pricing**: xAI Grok-beta offers excellent value
- **No Vendor Lock-in**: Easy to switch between different LLM providers

## Installation Guide

See [INSTALL.md](docs/INSTALL.md) for detailed setup instructions.

## API Documentation

See [docs/API.md](docs/API.md) for programmatic usage.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
```

### requirements.txt
```
langchain>=0.1.0
langchain-community>=0.0.12
faiss-cpu>=1.7.4
psycopg2-binary>=2.9.5
pandas>=1.5.0
numpy>=1.24.0
python-dotenv>=1.0.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.2
requests>=2.28.0
argparse>=1.4.0
```

### .env.example
```
# xAI Configuration
XAI_API_KEY=your_xai_api_key_here
XAI_BASE_URL=https://api.x.ai/v1

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gendb
DB_USER=genuser
DB_PASSWORD=genuser
DB_SCHEMA=genschema

# System Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=grok-beta
MAX_SIMILAR_TICKETS=10
CONFIDENCE_THRESHOLD=0.7
```

### setup.py
```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="smt-ticket-resolution-xai",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Intelligent SMT ticket resolution system using vector search and xAI Grok",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smt-ticket-resolution-xai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "smt-resolve=main:main",
        ],
    },
)
```

### config/__init__.py
```python
"""Configuration package for SMT Ticket Resolution System"""
```

### config/database.py
```python
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseConfig:
    HOST = os.getenv('DB_HOST', 'localhost')
    PORT = os.getenv('DB_PORT', '5432')
    NAME = os.getenv('DB_NAME', 'gendb')
    USER = os.getenv('DB_USER', 'genuser')
    PASSWORD = os.getenv('DB_PASSWORD', 'genuser')
    SCHEMA = os.getenv('DB_SCHEMA', 'genschema')
    
    @classmethod
    def get_connection_string(cls):
        return f"postgresql://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.NAME}"

class XAIConfig:
    API_KEY = os.getenv('XAI_API_KEY')
    BASE_URL = os.getenv('XAI_BASE_URL', 'https://api.x.ai/v1')
    MODEL_NAME = os.getenv('LLM_MODEL', 'grok-beta')
    
class EmbeddingConfig:
    MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    DEVICE = 'cpu'  # Change to 'cuda' if you have GPU
```

### models/__init__.py
```python
"""Models package for SMT Ticket Resolution System"""
```

### models/database_manager.py
```python
import psycopg2
import pandas as pd
from typing import List, Dict, Optional
from config.database import DatabaseConfig

class DatabaseManager:
    def __init__(self):
        self.config = DatabaseConfig()
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.config.HOST,
                port=self.config.PORT,
                database=self.config.NAME,
                user=self.config.USER,
                password=self.config.PASSWORD
            )
            print("Database connection established successfully!")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
    
    def get_all_tickets_with_solutions(self) -> pd.DataFrame:
        """Fetch all resolved tickets with their solutions"""
        query = f"""
        SELECT 
            t.ticket_id,
            t.machine_id,
            m.machine_name,
            m.location,
            t.operator_id,
            o.name as operator_name,
            o.shift,
            t.issue_description,
            t.status,
            t.priority,
            t.created_at,
            t.updated_at,
            ts.solution_id,
            ts.engineer_id,
            e.name as engineer_name,
            e.specialization,
            ts.solution_text,
            ts.resolved_at
        FROM {self.config.SCHEMA}.tickets t
        LEFT JOIN {self.config.SCHEMA}.machines m ON t.machine_id = m.machine_id
        LEFT JOIN {self.config.SCHEMA}.operators o ON t.operator_id = o.operator_id
        LEFT JOIN {self.config.SCHEMA}.ticket_solutions ts ON t.ticket_id = ts.ticket_id
        LEFT JOIN {self.config.SCHEMA}.engineers e ON ts.engineer_id = e.engineer_id
        WHERE t.status = 'Closed' AND ts.solution_text IS NOT NULL
        ORDER BY t.ticket_id;
        """
        
        try:
            df = pd.read_sql_query(query, self.connection)
            print(f"Retrieved {len(df)} resolved tickets with solutions")
            return df
        except Exception as e:
            print(f"Error fetching tickets: {e}")
            return pd.DataFrame()
    
    def get_ticket_by_id(self, ticket_id: int) -> Dict:
        """Get specific ticket details"""
        query = f"""
        SELECT 
            t.ticket_id,
            t.machine_id,
            m.machine_name,
            m.location,
            t.operator_id,
            o.name as operator_name,
            o.shift,
            t.issue_description,
            t.status,
            t.priority,
            t.created_at,
            t.updated_at
        FROM {self.config.SCHEMA}.tickets t
        LEFT JOIN {self.config.SCHEMA}.machines m ON t.machine_id = m.machine_id
        LEFT JOIN {self.config.SCHEMA}.operators o ON t.operator_id = o.operator_id
        WHERE t.ticket_id = %s;
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, (ticket_id,))
            result = cursor.fetchone()
            columns = [desc[0] for desc in cursor.description]
            cursor.close()
            
            if result:
                return dict(zip(columns, result))
            return {}
        except Exception as e:
            print(f"Error fetching ticket {ticket_id}: {e}")
            return {}
```

### models/vector_store.py
```python
import numpy as np
import pandas as pd
import faiss
import pickle
import os
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from utils.preprocessing import TextPreprocessor
from config.database import EmbeddingConfig

class TicketVectorStore:
    """FAISS-based vector store for ticket embeddings using Sentence-Transformers"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or EmbeddingConfig.MODEL_NAME
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=EmbeddingConfig.DEVICE)
        self.index = None
        self.ticket_data = None
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.preprocessor = TextPreprocessor()
        print(f"Embedding dimension: {self.dimension}")
        
    def build_index(self, tickets_df: pd.DataFrame, save_path: str = "data/embeddings/"):
        """Build FAISS index from ticket data"""
        print("Building vector index from ticket data...")
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Prepare ticket contexts
        print("Creating ticket contexts...")
        ticket_contexts = []
        ticket_metadata = []
        
        for idx, row in tickets_df.iterrows():
            context = self.preprocessor.create_ticket_context(row)
            cleaned_context = self.preprocessor.clean_text(context)
            
            ticket_contexts.append(cleaned_context)
            
            # Store metadata for each ticket
            metadata = {
                'ticket_id': row['ticket_id'],
                'issue_description': row['issue_description'],
                'machine_name': row['machine_name'],
                'location': row['location'],
                'priority': row['priority'],
                'solution_text': row['solution_text'],
                'engineer_name': row['engineer_name'],
                'specialization': row['specialization'],
                'original_context': context
            }
            ticket_metadata.append(metadata)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(ticket_contexts)} tickets...")
        embeddings = self.model.encode(ticket_contexts, show_progress_bar=True)
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
        # Store metadata
        self.ticket_data = ticket_metadata
        
        # Save index and metadata
        faiss.write_index(self.index, os.path.join(save_path, "ticket_index.faiss"))
        with open(os.path.join(save_path, "ticket_metadata.pkl"), 'wb') as f:
            pickle.dump(self.ticket_data, f)
        
        # Save model info
        model_info = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'total_tickets': len(ticket_contexts)
        }
        with open(os.path.join(save_path, "model_info.pkl"), 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"Vector index built successfully! {self.index.ntotal} vectors indexed.")
        
    def load_index(self, load_path: str = "data/embeddings/"):
        """Load existing FAISS index and metadata"""
        try:
            index_path = os.path.join(load_path, "ticket_index.faiss")
            metadata_path = os.path.join(load_path, "ticket_metadata.pkl")
            model_info_path = os.path.join(load_path, "model_info.pkl")
            
            if not all(os.path.exists(p) for p in [index_path, metadata_path, model_info_path]):
                print("Index files not found. Please build the index first.")
                return False
            
            # Load model info and verify compatibility
            with open(model_info_path, 'rb') as f:
                model_info = pickle.load(f)
            
            if model_info['model_name'] != self.model_name:
                print(f"Model mismatch. Index built with {model_info['model_name']}, "
                      f"but current model is {self.model_name}")
                return False
            
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.ticket_data = pickle.load(f)
            
            print(f"Vector index loaded successfully! {self.index.ntotal} vectors available.")
            print(f"Model: {model_info['model_name']}, Dimension: {model_info['dimension']}")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search_similar_tickets(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar tickets using vector similarity"""
        if self.index is None or self.ticket_data is None:
            print("Index not loaded. Please load or build the index first.")
            return []
        
        try:
            # Preprocess and embed the query
            cleaned_query = self.preprocessor.clean_text(query)
            query_embedding = self.model.encode([cleaned_query])
            query_vector = np.array(query_embedding).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = self.index.search(query_vector, k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.ticket_data):
                    result = self.ticket_data[idx].copy()
                    result['similarity_score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching similar tickets: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the vector index"""
        if self.index is None:
            return {"status": "Index not loaded"}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "index_type": type(self.index).__name__,
            "is_trained": self.index.is_trained,
            "model_name": self.model_name
        }
```

### models/ticket_resolver.py
```python
import requests
import json
from typing import List, Dict, Optional
from models.database_manager import DatabaseManager
from models.vector_store import TicketVectorStore
from config.database import XAIConfig

class XAIClient:
    """Client for xAI API"""
    
    def __init__(self):
        self.api_key = XAIConfig.API_KEY
        self.base_url = XAIConfig.BASE_URL
        self.model_name = XAIConfig.MODEL_NAME
        
        if not self.api_key:
            raise ValueError("XAI_API_KEY environment variable is required")
    
    def chat_completion(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Make a chat completion request to xAI"""
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling xAI API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return f"Error generating recommendation: {e}"
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            return f"Error parsing API response: {e}"

class SMTTicketResolver:
    """Main class for intelligent ticket resolution using xAI Grok"""
    
    def __init__(self):
        self.xai_client = XAIClient()
        self.db_manager = DatabaseManager()
        self.vector_store = TicketVectorStore()
        
    def initialize(self, rebuild_index: bool = False):
        """Initialize the system"""
        print("Initializing SMT Ticket Resolution System with xAI Grok...")
        
        # Connect to database
        if not self.db_manager.connect():
            return False
        
        # Load or build vector index
        if rebuild_index or not self.vector_store.load_index():
            print("Building new vector index...")
            tickets_df = self.db_manager.get_all_tickets_with_solutions()
            if tickets_df.empty:
                print("No resolved tickets found to build index.")
                return False
            self.vector_store.build_index(tickets_df)
        
        print("System initialized successfully!")
        return True
    
    def resolve_ticket(self, issue_description: str, machine_name: str = None, 
                      priority: str = None, k_similar: int = 5) -> Dict:
        """Get resolution recommendation for a new ticket"""
        
        # Create query context
        query_parts = [f"Issue: {issue_description}"]
        if machine_name:
            query_parts.append(f"Machine: {machine_name}")
        if priority:
            query_parts.append(f"Priority: {priority}")
        
        query = " | ".join(query_parts)
        
        # Find similar tickets
        similar_tickets = self.vector_store.search_similar_tickets(query, k=k_similar)
        
        if not similar_tickets:
            return {
                "status": "error",
                "message": "No similar tickets found",
                "recommendation": "Please consult with an SMT engineer for manual diagnosis."
            }
        
        # Generate Grok recommendation
        recommendation = self._generate_recommendation(
            issue_description, machine_name, priority, similar_tickets
        )
        
        return {
            "status": "success",
            "query": query,
            "similar_tickets": similar_tickets,
            "recommendation": recommendation,
            "confidence_score": self._calculate_confidence(similar_tickets)
        }
    
    def _generate_recommendation(self, issue_description: str, machine_name: str,
                                priority: str, similar_tickets: List[Dict]) -> str:
        """Generate xAI Grok-based resolution recommendation"""
        
        # Prepare similar tickets context
        similar_context = ""
        for i, ticket in enumerate(similar_tickets[:3], 1):  # Top 3 tickets
            similar_context += f"""
            Similar Ticket #{i} (Similarity: {ticket['similarity_score']:.3f}):
            - Issue: {ticket['issue_description']}
            - Machine: {ticket['machine_name']}
            - Priority: {ticket['priority']}
            - Solution: {ticket['solution_text']}
            - Engineer: {ticket['engineer_name']} ({ticket['specialization']})
            """
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert SMT (Surface Mount Technology) engineer with years of experience in troubleshooting PCB assembly issues. Your role is to analyze new tickets and provide actionable resolution recommendations based on similar past cases.

SMT Context:
- Pick and Place: Component placement machines that can have nozzle, feeder, or vision issues
- Reflow Oven: Temperature control for soldering, can have profile or heating issues  
- AOI (Automated Optical Inspection): Vision systems that can have false positives or calibration issues
- SPI (Solder Paste Inspection): Measures paste height and volume, can have calibration issues
- Common Issues: Nozzle pickup errors, component misalignment, solder bridging, tombstoning, PCB warping, feeder jams

Provide practical, step-by-step solutions based on the similar tickets provided. Be concise but comprehensive."""
            },
            {
                "role": "user",
                "content": f"""
NEW TICKET:
Issue Description: {issue_description}
Machine: {machine_name or 'Not specified'}
Priority: {priority or 'Not specified'}

SIMILAR RESOLVED TICKETS:
{similar_context}

Based on the similar tickets above, provide a detailed resolution recommendation for this new ticket. Include:
1. Root cause analysis
2. Step-by-step solution
3. Preventive measures
4. Estimated resolution time
5. Required expertise level

Format your response as a clear, actionable recommendation that a technician can follow.
"""
            }
        ]
        
        return self.xai_client.chat_completion(messages)
    
    def _calculate_confidence(self, similar_tickets: List[Dict]) -> float:
        """Calculate confidence score based on similarity scores"""
        if not similar_tickets:
            return 0.0
        
        # Average of top 3 similarity scores
        top_scores = [ticket['similarity_score'] for ticket in similar_tickets[:3]]
        confidence = sum(top_scores) / len(top_scores)
        
        # Convert to percentage
        return round(confidence * 100, 2)
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            "database_connected": self.db_manager.connection is not None,
            "vector_index_stats": self.vector_store.get_index_stats(),
            "xai_model": self.xai_client.model_name
        }
        
        if stats["database_connected"]:
            # Get some database stats
            try:
                query = f"SELECT COUNT(*) as total_tickets FROM {self.db_manager.config.SCHEMA}.tickets"
                cursor = self.db_manager.connection.cursor()
                cursor.execute(query)
                stats["total_tickets"] = cursor.fetchone()[0]
                cursor.close()
                
                query = f"SELECT COUNT(*) as resolved_tickets FROM {self.db_manager.config.SCHEMA}.tickets WHERE status = 'Closed'"
                cursor = self.db_manager.connection.cursor()
                cursor.execute(query)
                stats["resolved_tickets"] = cursor.fetchone()[0]
                cursor.close()
                
            except Exception as e:
                stats["database_error"] = str(e)
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        self.db_manager.disconnect()
```

### utils/__init__.py
```python
"""Utilities package for SMT Ticket Resolution System"""
```

### utils/preprocessing.py
```python
import re
import string
import pandas as pd
from typing import List

class TextPreprocessor:
    """Handles text cleaning and preprocessing for SMT ticket data"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text for better embedding quality"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep meaningful punctuation
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def create_ticket_context(row: pd.Series) -> str:
        """Create rich context from ticket data for better embeddings"""
        context_parts = []
        
        # Issue description (most important)
        if row.get('issue_description'):
            context_parts.append(f"Issue: {row['issue_description']}")
        
        # Machine information
        if row.get('machine_name'):
            context_parts.append(f"Machine: {row['machine_name']}")
        
        if row.get('location'):
            context_parts.append(f"Location: {row['location']}")
        
        # Priority and status
        if row.get('priority'):
            context_parts.append(f"Priority: {row['priority']}")
        
        # Operator shift (can indicate complexity)
        if row.get('shift'):
            context_parts.append(f"Shift: {row['shift']}")
        
        # Solution (for resolved tickets)
        if row.get('solution_text'):
            context_parts.append(f"Solution: {row['solution_text']}")
        
        # Engineer specialization
        if row.get('specialization'):
            context_parts.append(f"Specialist: {row['specialization']}")
        
        return " | ".join(context_parts)
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """Extract important keywords from ticket text"""
        # SMT-specific technical terms
        smt_keywords = [
            'nozzle', 'pickup', 'placement', 'feeder', 'jam', 'solder', 'paste',
            'bridging', 'reflow', 'temperature', 'oven', 'aoi', 'spi', 'height',
            'deviation', 'component', 'misalignment', 'tombstoning', 'pcb',
            'warping', 'false positive', 'detection', 'calibration', 'vision'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in smt_keywords if kw in text_lower]
        
        return found_keywords
```

# SMT Ticket Resolution System - xAI Grok Edition (Part 2)

## Continuing from Part 1...

### setup_system.py (continued)
```python
#!/usr/bin/env python3
"""
Setup script for SMT Ticket Resolution System with xAI
"""

import os
import sys
from models.database_manager import DatabaseManager
from models.vector_store import TicketVectorStore
from models.ticket_resolver import SMTTicketResolver

def setup_system():
    """Complete system setup"""
    print("=== SMT Ticket Resolution System Setup (xAI Edition) ===\n")
    
    # Check environment variables
    print("1. Checking environment variables...")
    required_vars = ['XAI_API_KEY', 'DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the required variables.")
        print("Get your xAI API key from: https://console.x.ai/")
        return False
    print("✅ Environment variables configured")
    
    # Test database connection
    print("\n2. Testing database connection...")
    db_manager = DatabaseManager()
    if not db_manager.connect():
        print("❌ Database connection failed")
        return False
    print("✅ Database connection successful")
    
    # Fetch tickets data
    print("\n3. Fetching resolved tickets...")
    tickets_df = db_manager.get_all_tickets_with_solutions()
    if tickets_df.empty:
        print("❌ No resolved tickets found in database")
        db_manager.disconnect()
        return False
    print(f"✅ Found {len(tickets_df)} resolved tickets")
    
    # Build vector index
    print("\n4. Building vector index with Sentence-Transformers...")
    print("   (This will download the embedding model on first run)")
    vector_store = TicketVectorStore()
    try:
        vector_store.build_index(tickets_df)
        print("✅ Vector index built successfully")
    except Exception as e:
        print(f"❌ Failed to build vector index: {e}")
        db_manager.disconnect()
        return False
    
    # Test the system
    print("\n5. Testing system with sample query...")
    resolver = SMTTicketResolver()
    if not resolver.initialize():
        print("❌ System initialization failed")
        return False
    
    # Test with a sample issue
    test_result = resolver.resolve_ticket(
        issue_description="Nozzle pickup error on component placement",
        machine_name="Pick and Place Line 1",
        priority="Medium"
    )
    
    if test_result["status"] == "success":
        print("✅ System test successful")
        print(f"   - Found {len(test_result['similar_tickets'])} similar tickets")
        print(f"   - Confidence score: {test_result['confidence_score']}%")
        print(f"   - Using xAI Grok model: {resolver.xai_client.model_name}")
    else:
        print(f"❌ System test failed: {test_result.get('message', 'Unknown error')}")
        resolver.cleanup()
        return False
    
    # Cleanup
    resolver.cleanup()
    
    print("\n🎉 SMT Ticket Resolution System setup completed successfully!")
    print("\nYou can now use main.py to resolve tickets.")
    return True

if __name__ == "__main__":
    if setup_system():
        sys.exit(0)
    else:
        sys.exit(1)
```

### main.py
```python
#!/usr/bin/env python3
"""
Main application for SMT Ticket Resolution System with xAI Grok
"""

import argparse
import json
from datetime import datetime
from models.ticket_resolver import SMTTicketResolver

class TicketResolutionCLI:
    """Command Line Interface for the ticket resolution system"""
    
    def __init__(self):
        self.resolver = SMTTicketResolver()
        
    def run_interactive_mode(self):
        """Interactive mode for resolving tickets"""
        print("=== SMT Ticket Resolution System (xAI Grok Edition) ===")
        print("Interactive mode - Type 'exit' to quit\n")
        
        if not self.resolver.initialize():
            print("Failed to initialize system. Please run setup_system.py first.")
            return
        
        # Display system info
        stats = self.resolver.get_system_stats()
        print(f"🤖 Using xAI Model: {stats.get('xai_model', 'Unknown')}")
        print(f"📊 Vector Index: {stats['vector_index_stats'].get('total_vectors', 0)} tickets indexed")
        print(f"🔧 Embedding Model: {stats['vector_index_stats'].get('model_name', 'Unknown')}")
        
        while True:
            print("\n" + "="*50)
            issue = input("Enter issue description (or 'exit'): ").strip()
            
            if issue.lower() == 'exit':
                break
                
            if not issue:
                print("Please enter a valid issue description.")
                continue
            
            machine = input("Enter machine name (optional): ").strip() or None
            priority = input("Enter priority (Low/Medium/High, optional): ").strip() or None
            
            print("\n🔍 Searching for similar tickets...")
            result = self.resolver.resolve_ticket(issue, machine, priority)
            
            if result["status"] == "success":
                self.display_results(result)
            else:
                print(f"❌ Error: {result['message']}")
        
        self.resolver.cleanup()
        print("\nGoodbye!")
    
    def resolve_single_ticket(self, issue: str, machine: str = None, priority: str = None):
        """Resolve a single ticket from command line arguments"""
        print("=== SMT Ticket Resolution System (xAI Grok Edition) ===")
        
        if not self.resolver.initialize():
            print("Failed to initialize system. Please run setup_system.py first.")
            return
        
        print(f"\n🔍 Resolving ticket: {issue}")
        result = self.resolver.resolve_ticket(issue, machine, priority)
        
        if result["status"] == "success":
            self.display_results(result)
        else:
            print(f"❌ Error: {result['message']}")
        
        self.resolver.cleanup()
    
    def display_results(self, result: dict):
        """Display resolution results in a formatted way"""
        print(f"\n✅ Found {len(result['similar_tickets'])} similar tickets")
        print(f"🎯 Confidence Score: {result['confidence_score']}%")
        
        print("\n📋 SIMILAR TICKETS:")
        print("-" * 60)
        
        for i, ticket in enumerate(result['similar_tickets'][:3], 1):
            print(f"\n{i}. Ticket #{ticket['ticket_id']} (Similarity: {ticket['similarity_score']:.3f})")
            print(f"   Issue: {ticket['issue_description']}")
            print(f"   Machine: {ticket['machine_name']}")
            print(f"   Solution: {ticket['solution_text']}")
            print(f"   Engineer: {ticket['engineer_name']} ({ticket['specialization']})")
        
        print("\n🤖 GROK'S RECOMMENDED RESOLUTION:")
        print("=" * 60)
        print(result['recommendation'])
        print("=" * 60)
    
    def show_stats(self):
        """Display system statistics"""
        print("=== SMT Ticket Resolution System Stats ===")
        
        if not self.resolver.initialize():
            print("Failed to initialize system.")
            return
        
        stats = self.resolver.get_system_stats()
        
        print(f"\n🤖 AI Model: {stats.get('xai_model', 'Unknown')}")
        print(f"🗄️  Database Connected: {stats['database_connected']}")
        
        if 'total_tickets' in stats:
            print(f"📊 Total Tickets: {stats['total_tickets']}")
            print(f"✅ Resolved Tickets: {stats['resolved_tickets']}")
        
        vector_stats = stats['vector_index_stats']
        if vector_stats.get('status') != 'Index not loaded':
            print(f"\n🔍 Vector Index Stats:")
            print(f"   - Total Vectors: {vector_stats['total_vectors']}")
            print(f"   - Embedding Model: {vector_stats['model_name']}")
            print(f"   - Dimension: {vector_stats['dimension']}")
            print(f"   - Index Type: {vector_stats['index_type']}")
        
        self.resolver.cleanup()

def main():
    parser = argparse.ArgumentParser(description="SMT Ticket Resolution System with xAI Grok")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--issue", type=str, 
                       help="Issue description for single ticket resolution")
    parser.add_argument("--machine", type=str, 
                       help="Machine name (optional)")
    parser.add_argument("--priority", type=str, choices=["Low", "Medium", "High"],
                       help="Priority level (optional)")
    parser.add_argument("--stats", action="store_true",
                       help="Show system statistics")
    
    args = parser.parse_args()
    
    cli = TicketResolutionCLI()
    
    if args.stats:
        cli.show_stats()
    elif args.interactive:
        cli.run_interactive_mode()
    elif args.issue:
        cli.resolve_single_ticket(args.issue, args.machine, args.priority)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

### tests/__init__.py
```python
"""Tests package for SMT Ticket Resolution System"""
```

### tests/test_system.py
```python
"""
Test suite for SMT Ticket Resolution System
"""

import unittest
import os
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock

from models.database_manager import DatabaseManager
from models.vector_store import TicketVectorStore
from models.ticket_resolver import SMTTicketResolver, XAIClient
from utils.preprocessing import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    """Test text preprocessing functionality"""
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text(self):
        """Test text cleaning"""
        text = "  NOZZLE pickup ERROR!!! Multiple    spaces  "
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "nozzle pickup error multiple spaces")
    
    def test_empty_text(self):
        """Test handling of empty text"""
        self.assertEqual(self.preprocessor.clean_text(""), "")
        self.assertEqual(self.preprocessor.clean_text(None), "")
    
    def test_extract_keywords(self):
        """Test SMT keyword extraction"""
        text = "Nozzle pickup error on reflow oven with solder bridging"
        keywords = self.preprocessor.extract_keywords(text)
        expected = ['nozzle', 'pickup', 'reflow', 'oven', 'solder', 'bridging']
        self.assertEqual(set(keywords), set(expected))

class TestVectorStore(unittest.TestCase):
    """Test vector store functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = TicketVectorStore()
        
        # Create sample data
        self.sample_data = pd.DataFrame([
            {
                'ticket_id': 1,
                'issue_description': 'Nozzle pickup error',
                'machine_name': 'Pick and Place Line 1',
                'location': 'Line A',
                'priority': 'Medium',
                'solution_text': 'Replaced faulty nozzle',
                'engineer_name': 'John Doe',
                'specialization': 'SMT Calibration'
            },
            {
                'ticket_id': 2,
                'issue_description': 'Solder bridging observed',
                'machine_name': 'Reflow Oven',
                'location': 'Line B',
                'priority': 'High',
                'solution_text': 'Adjusted temperature profile',
                'engineer_name': 'Jane Smith',
                'specialization': 'Thermal Profiling'
            }
        ])
    
    def test_create_ticket_context(self):
        """Test ticket context creation"""
        preprocessor = TextPreprocessor()
        context = preprocessor.create_ticket_context(self.sample_data.iloc[0])
        
        self.assertIn("Nozzle pickup error", context)
        self.assertIn("Pick and Place Line 1", context)
        self.assertIn("Medium", context)
        self.assertIn("Replaced faulty nozzle", context)
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_build_index(self, mock_model):
        """Test index building"""
        # Mock the sentence transformer
        mock_model_instance = MagicMock()
        mock_model_instance.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model_instance.get_sentence_embedding_dimension.return_value = 3
        mock_model.return_value = mock_model_instance
        
        # Test building index
        try:
            self.vector_store.build_index(self.sample_data, self.temp_dir)
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"build_index raised an exception: {e}")

class TestXAIClient(unittest.TestCase):
    """Test xAI client functionality"""
    
    def setUp(self):
        # Mock environment variable
        os.environ['XAI_API_KEY'] = 'test_key'
        self.client = XAIClient()
    
    @patch('requests.post')
    def test_chat_completion_success(self, mock_post):
        """Test successful chat completion"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test message"}]
        result = self.client.chat_completion(messages)
        
        self.assertEqual(result, "Test response")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_chat_completion_error(self, mock_post):
        """Test chat completion with API error"""
        # Mock error response
        mock_post.side_effect = Exception("API Error")
        
        messages = [{"role": "user", "content": "Test message"}]
        result = self.client.chat_completion(messages)
        
        self.assertIn("Error generating recommendation", result)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    @patch.dict(os.environ, {
        'XAI_API_KEY': 'test_key',
        'DB_HOST': 'localhost',
        'DB_NAME': 'test_db',
        'DB_USER': 'test_user',
        'DB_PASSWORD': 'test_pass'
    })
    def test_resolver_initialization(self):
        """Test resolver initialization"""
        resolver = SMTTicketResolver()
        
        # Test that components are properly initialized
        self.assertIsNotNone(resolver.xai_client)
        self.assertIsNotNone(resolver.db_manager)
        self.assertIsNotNone(resolver.vector_store)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        resolver = SMTTicketResolver()
        
        similar_tickets = [
            {'similarity_score': 0.9},
            {'similarity_score': 0.8},
            {'similarity_score': 0.7}
        ]
        
        confidence = resolver._calculate_confidence(similar_tickets)
        expected = round(((0.9 + 0.8 + 0.7) / 3) * 100, 2)
        self.assertEqual(confidence, expected)

if __name__ == '__main__':
    unittest.main()
```

### docs/API.md
```markdown
# SMT Ticket Resolution System API Documentation

## Overview

The SMT Ticket Resolution System provides a programmatic interface for resolving manufacturing tickets using vector similarity search and xAI Grok.

## Core Classes

### SMTTicketResolver

Main class for ticket resolution functionality.

```python
from models.ticket_resolver import SMTTicketResolver

resolver = SMTTicketResolver()
resolver.initialize()

result = resolver.resolve_ticket(
    issue_description="Nozzle pickup error",
    machine_name="Pick and Place Line 1",
    priority="Medium"
)
```

#### Methods

**`initialize(rebuild_index=False)`**
- Initializes the system
- `rebuild_index`: Force rebuild of vector index
- Returns: `bool` - Success status

**`resolve_ticket(issue_description, machine_name=None, priority=None, k_similar=5)`**
- Resolves a ticket and returns recommendations
- `issue_description`: Description of the issue
- `machine_name`: Optional machine name
- `priority`: Optional priority (Low/Medium/High)
- `k_similar`: Number of similar tickets to retrieve
- Returns: `dict` with resolution results

**`get_system_stats()`**
- Returns system statistics
- Returns: `dict` with system information

**`cleanup()`**
- Cleans up resources and database connections

### Response Format

```python
{
    "status": "success",
    "query": "Issue: Nozzle pickup error | Machine: Pick and Place Line 1",
    "similar_tickets": [
        {
            "ticket_id": 123,
            "issue_description": "Nozzle pickup error",
            "machine_name": "Pick and Place Line 1",
            "solution_text": "Replaced faulty nozzle",
            "similarity_score": 0.95,
            "rank": 1
        }
    ],
    "recommendation": "Detailed Grok-generated recommendation...",
    "confidence_score": 87.5
}
```

### XAIClient

Client for interacting with xAI's API.

```python
from models.ticket_resolver import XAIClient

client = XAIClient()
response = client.chat_completion([
    {"role": "user", "content": "Analyze this SMT issue..."}
])
```

### TicketVectorStore

Vector store for similarity search.

```python
from models.vector_store import TicketVectorStore

store = TicketVectorStore()
store.load_index()
results = store.search_similar_tickets("nozzle error", k=5)
```

## Configuration

### Environment Variables

```bash
# xAI Configuration
XAI_API_KEY=your_xai_api_key
XAI_BASE_URL=https://api.x.ai/v1

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gendb
DB_USER=genuser
DB_PASSWORD=genuser
DB_SCHEMA=genschema

# System Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=grok-beta
```

## Usage Examples

### Basic Resolution

```python
from models.ticket_resolver import SMTTicketResolver

resolver = SMTTicketResolver()
if resolver.initialize():
    result = resolver.resolve_ticket("Component misalignment detected")
    print(result['recommendation'])
    resolver.cleanup()
```

### Batch Processing

```python
issues = [
    {"description": "Nozzle pickup error", "machine": "PnP Line 1"},
    {"description": "Solder bridging", "machine": "Reflow Oven"},
    {"description": "AOI false positive", "machine": "AOI Machine"}
]

resolver = SMTTicketResolver()
resolver.initialize()

for issue in issues:
    result = resolver.resolve_ticket(
        issue["description"], 
        issue["machine"]
    )
    print(f"Issue: {issue['description']}")
    print(f"Confidence: {result['confidence_score']}%")
    print(f"Recommendation: {result['recommendation'][:100]}...")
    print("-" * 50)

resolver.cleanup()
```

### Custom Similarity Threshold

```python
result = resolver.resolve_ticket(
    "Temperature deviation in reflow",
    k_similar=10  # Get more similar tickets
)

# Filter by confidence
if result['confidence_score'] > 80:
    print("High confidence recommendation:")
    print(result['recommendation'])
else:
    print("Low confidence - consider manual review")
```

## Error Handling

```python
try:
    resolver = SMTTicketResolver()
    if not resolver.initialize():
        raise Exception("Failed to initialize system")
    
    result = resolver.resolve_ticket("Test issue")
    
    if result['status'] == 'error':
        print(f"Resolution failed: {result['message']}")
    else:
        print(f"Found {len(result['similar_tickets'])} similar tickets")
        
except Exception as e:
    print(f"System error: {e}")
finally:
    if 'resolver' in locals():
        resolver.cleanup()
```

## Performance Considerations

- **Index Loading**: Load index once and reuse for multiple queries
- **Batch Processing**: Initialize once for multiple ticket resolutions
- **Memory Usage**: Vector index loads into memory (~100MB for 1000 tickets)
- **API Limits**: xAI has rate limits - implement appropriate delays for high-volume usage

## Customization

### Custom Embedding Model

```python
from models.vector_store import TicketVectorStore

# Use a different Sentence-Transformers model
store = TicketVectorStore(model_name="all-mpnet-base-v2")
```

### Custom Preprocessing

```python
from utils.preprocessing import TextPreprocessor

class CustomPreprocessor(TextPreprocessor):
    @staticmethod
    def clean_text(text):
        # Custom cleaning logic
        return text.lower().strip()

# Use in vector store
store.preprocessor = CustomPreprocessor()
```
```

### INSTALL.md
```markdown
# Installation Guide - SMT Ticket Resolution System (xAI Edition)

## Prerequisites

- Python 3.8 or higher
- PostgreSQL database with SMT ticket data
- xAI API key (get from https://console.x.ai/)
- 4GB+ RAM (for embedding model)

## Step 1: Environment Setup

### Create Virtual Environment
```bash
python -m venv smt_env
source smt_env/bin/activate  # On Windows: smt_env\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Configuration

### Create Environment File
```bash
cp .env.example .env
```

### Edit .env File
```bash
# xAI Configuration
XAI_API_KEY=your_actual_xai_api_key_here
XAI_BASE_URL=https://api.x.ai/v1

# Database Configuration
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=gendb
DB_USER=genuser
DB_PASSWORD=genuser
DB_SCHEMA=genschema
```

## Step 3: Database Setup

Ensure your PostgreSQL database contains the SMT ticket schema. If using the provided schema:

```sql
-- Connect to your database and run the schema creation script
\i smt_ticket_full_system.sql
```

## Step 4: System Initialization

### Run Setup Script
```bash
python setup_system.py
```

This will:
- Verify environment variables
- Test database connection
- Download embedding model (first run only)
- Build vector index from your ticket data
- Test xAI integration

## Step 5: Verification

### Test Interactive Mode
```bash
python main.py --interactive
```

### Test Single Query
```bash
python main.py --issue "Nozzle pickup error" --machine "Pick and Place Line 1"
```

### Check System Stats
```bash
python main.py --stats
```

## Troubleshooting

### Common Issues

**1. xAI API Key Issues**
```bash
Error: XAI_API_KEY environment variable is required
```
- Verify your API key is correct
- Check .env file is in the project root
- Ensure no extra spaces in the API key

**2. Database Connection Issues**
```bash
Error connecting to database: FATAL: password authentication failed
```
- Verify database credentials in .env
- Ensure PostgreSQL is running
- Check network connectivity to database

**3. Embedding Model Download Issues**
```bash
Error downloading model: Connection timeout
```
- Ensure internet connection
- Try again - downloads can be interrupted
- Model is cached after first download

**4. Memory Issues**
```bash
Out of memory error during index building
```
- Reduce batch size in vector store
- Ensure 4GB+ RAM available
- Close other applications

### Performance Optimization

**For Large Datasets (10k+ tickets):**
```python
# Modify vector_store.py
embeddings = self.model.encode(
    ticket_contexts, 
    batch_size=32,  # Reduce from default
    show_progress_bar=True
)
```

**For Low Memory Systems:**
```python
# Use smaller embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2  # 80MB model
# Instead of: all-mpnet-base-v2  # 420MB model
```

## Development Setup

### Install Development Dependencies
```bash
pip install pytest black flake8 mypy
```

### Run Tests
```bash
python -m pytest tests/
```

### Code Formatting
```bash
black models/ utils/ tests/
```

## Production Deployment

### Docker Setup (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py", "--interactive"]
```

### Environment Variables for Production
```bash
# Use environment-specific values
export XAI_API_KEY="prod_key"
export DB_HOST="prod.db.server"
export DB_PASSWORD="secure_prod_password"
```

### Monitoring
- Monitor xAI API usage and costs
- Track vector index performance
- Log ticket resolution accuracy

## Next Steps

1. **Integration**: Add API endpoints for web integration
2. **Monitoring**: Implement logging and metrics
3. **Scaling**: Consider distributed vector storage
4. **Customization**: Adapt preprocessing for your specific SMT data

## Support

For issues:
1. Check logs in the console output
2. Verify environment configuration
3. Test individual components (DB, xAI, embeddings)
4. Review system stats with `--stats` flag
```

### data/embeddings/.gitkeep
```
# This file ensures the embeddings directory is created in git
# Vector index files will be stored here after running setup_system.py
```

### LICENSE
```
MIT License

Copyright (c) 2024 SMT Ticket Resolution System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
