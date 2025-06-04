# SMT Ticket Resolution System - Setup & Run Instructions

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- PostgreSQL database with SMT ticket data
- xAI API key from https://console.x.ai/
- 4GB+ RAM recommended

---

## 📦 Installation Steps

### 1. Download and Extract
```bash
# Download the complete package
# Extract to your desired directory
cd smt_ticket_resolution_xai/
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual values:
XAI_API_KEY=your_xai_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gendb
DB_USER=genuser
DB_PASSWORD=genuser
DB_SCHEMA=genschema
```

### 5. Setup Database
Make sure your PostgreSQL database is running and contains the SMT ticket schema. If you have the provided SQL file:

```sql
-- Connect to PostgreSQL and run:
\i smt_ticket_full_system.sql
```

### 6. Initialize System
```bash
python setup_system.py
```

This will:
- ✅ Verify environment variables
- 🔗 Test database connection  
- 📥 Download embedding model (first run only)
- 🏗️ Build vector index from ticket data
- 🧪 Test xAI Grok integration

---

## 🏃‍♂️ Running the System

### Interactive Mode (Recommended for Testing)
```bash
python main.py --interactive
```

Example session:
```
=== SMT Ticket Resolution System (xAI Grok Edition) ===
🤖 Using xAI Model: grok-beta
📊 Vector Index: 357 tickets indexed
🔧 Embedding Model: all-MiniLM-L6-v2

Enter issue description (or 'exit'): Nozzle pickup error
Enter machine name (optional): Pick and Place Line 1
Enter priority (Low/Medium/High, optional): Medium

🔍 Searching for similar tickets...
✅ Found 5 similar tickets
🎯 Confidence Score: 87.5%

📋 SIMILAR TICKETS:
1. Ticket #123 (Similarity: 0.945)