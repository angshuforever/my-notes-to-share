# Smart MES Machine Management System
## FastAPI & PostgreSQL Implementation Guide

This guide provides step-by-step instructions for setting up and implementing a CRUD (Create, Read, Update, Delete) application for machine management in a smart Manufacturing Execution System (MES) using FastAPI and PostgreSQL.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Database Setup](#database-setup)
- [Application Setup](#application-setup)
- [API Implementation](#api-implementation)
- [Testing the Application](#testing-the-application)
- [Sample API Requests](#sample-api-requests)

## Prerequisites

Ensure you have the following installed:
- Python 3.8+
- PostgreSQL 12+
- pip (Python package manager)

## Project Structure

```
mes_system/
├── requirements.txt
├── database.py
├── models.py
├── schemas.py
└── main.py
```

## Database Setup

### 1. Create Database and Table

Connect to PostgreSQL and run the following commands:

```sql
-- Create Database
CREATE DATABASE mes_db;

-- Connect to Database
\c mes_db

-- Create Table
CREATE TABLE machines (
    id SERIAL PRIMARY KEY,
    machine_name VARCHAR UNIQUE NOT NULL,
    machine_type VARCHAR NOT NULL,
    operational_status VARCHAR NOT NULL,
    efficiency FLOAT NOT NULL,
    last_maintenance TIMESTAMP NOT NULL
);
```

### 2. Insert Sample Data

```sql
-- Insert Sample Data
INSERT INTO machines (machine_name, machine_type, operational_status, efficiency, last_maintenance)
VALUES 
    ('CNC-001', 'Milling Machine', 'Running', 85.5, '2024-01-10 08:00:00'),
    ('ROB-002', 'Robot Arm', 'Idle', 92.0, '2024-01-09 14:30:00'),
    ('ASM-003', 'Assembly Station', 'Maintenance', 78.3, '2024-01-11 10:15:00'),
    ('CNC-004', 'Lathe Machine', 'Down', 0.0, '2024-01-08 16:45:00');
```

## Application Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

Create `requirements.txt`:
```txt
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
pydantic==2.5.2
python-dotenv==1.0.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Database Configuration (database.py)

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

POSTGRES_URL = "postgresql://username:password@localhost:5432/mes_db"

engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

### 4. Define Models (models.py)

```python
from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base

class Machine(Base):
    __tablename__ = "machines"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_name = Column(String, unique=True, index=True)
    machine_type = Column(String)
    operational_status = Column(String)  # Running, Idle, Maintenance, Down
    efficiency = Column(Float)  # OEE (Overall Equipment Effectiveness)
    last_maintenance = Column(DateTime)
```

### 5. Define Schemas (schemas.py)

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class MachineBase(BaseModel):
    machine_name: str
    machine_type: str
    operational_status: str
    efficiency: float
    last_maintenance: datetime

class MachineCreate(MachineBase):
    pass

class Machine(MachineBase):
    id: int
    
    class Config:
        orm_mode = True
```

### 6. Implement API (main.py)

```python
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from database import SessionLocal, engine
import models, schemas
from datetime import datetime

app = FastAPI(title="Smart MES Machine Management")
models.Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/machines/", response_model=schemas.Machine)
def create_machine(machine: schemas.MachineCreate, db: Session = Depends(get_db)):
    db_machine = models.Machine(**machine.dict())
    db.add(db_machine)
    db.commit()
    db.refresh(db_machine)
    return db_machine

@app.get("/machines/", response_model=List[schemas.Machine])
def read_machines(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    machines = db.query(models.Machine).offset(skip).limit(limit).all()
    return machines

@app.get("/machines/{machine_id}", response_model=schemas.Machine)
def read_machine(machine_id: int, db: Session = Depends(get_db)):
    machine = db.query(models.Machine).filter(models.Machine.id == machine_id).first()
    if machine is None:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine

@app.put("/machines/{machine_id}", response_model=schemas.Machine)
def update_machine(machine_id: int, machine: schemas.MachineCreate, db: Session = Depends(get_db)):
    db_machine = db.query(models.Machine).filter(models.Machine.id == machine_id).first()
    if db_machine is None:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    for key, value in machine.dict().items():
        setattr(db_machine, key, value)
    
    db.commit()
    db.refresh(db_machine)
    return db_machine

@app.delete("/machines/{machine_id}")
def delete_machine(machine_id: int, db: Session = Depends(get_db)):
    machine = db.query(models.Machine).filter(models.Machine.id == machine_id).first()
    if machine is None:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    db.delete(machine)
    db.commit()
    return {"message": "Machine deleted successfully"}
```

## Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`
Interactive API documentation will be available at `http://localhost:8000/docs`

## Sample API Requests

### Create Machine
```bash
curl -X 'POST' \
  'http://localhost:8000/machines/' \
  -H 'Content-Type: application/json' \
  -d '{
  "machine_name": "CNC-005",
  "machine_type": "Milling Machine",
  "operational_status": "Running",
  "efficiency": 88.5,
  "last_maintenance": "2024-01-12T08:00:00"
}'
```

### Get All Machines
```bash
curl -X 'GET' 'http://localhost:8000/machines/'
```

### Get Single Machine
```bash
curl -X 'GET' 'http://localhost:8000/machines/1'
```

### Update Machine
```bash
curl -X 'PUT' \
  'http://localhost:8000/machines/1' \
  -H 'Content-Type: application/json' \
  -d '{
  "machine_name": "CNC-001",
  "machine_type": "Milling Machine",
  "operational_status": "Maintenance",
  "efficiency": 0.0,
  "last_maintenance": "2024-01-12T14:30:00"
}'
```

### Delete Machine
```bash
curl -X 'DELETE' 'http://localhost:8000/machines/1'
```

## Error Handling

The API includes basic error handling for common scenarios:
- 404: Machine not found
- 422: Validation error (invalid data format)
- 500: Internal server error

## Data Model

The Machine entity includes these critical parameters:
1. `machine_name` (string): Unique identifier for the machine
2. `machine_type` (string): Type/category of the machine
3. `operational_status` (string): Current status (Running/Idle/Maintenance/Down)
4. `efficiency` (float): Overall Equipment Effectiveness (OEE)
5. `last_maintenance` (datetime): Timestamp of the last maintenance

## Security Considerations

For production deployment, consider implementing:
1. Authentication using JWT or OAuth2
2. Rate limiting
3. CORS policies
4. Environment variables for sensitive configuration
5. Input validation and sanitization
6. SSL/TLS encryption

---

This implementation provides a foundation for a machine management system. You can extend it by adding more features such as:
- Machine performance metrics
- Maintenance scheduling
- Alert notifications
- User authentication
- Audit logging
