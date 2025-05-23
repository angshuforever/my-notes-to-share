# Complete DuckDB Tutorial: From Installation to MotherDuck Integration



## Table of Contents

1. [Installation on Mac](#installation-on-mac)

2. [Basic DuckDB Operations](#basic-duckdb-operations)

3. [Working with Different File Formats](#working-with-different-file-formats)

4. [Persistent Storage](#persistent-storage)

5. [Python Integration](#python-integration)

6. [FastAPI Integration](#fastapi-integration)

7. [MotherDuck Integration](#motherduck-integration)

8. [Complete Example Project](#complete-example-project)



## Installation on Mac



### Step 1: Install DuckDB CLI

```bash

# Using Homebrew (recommended)

brew install duckdb



# Or download directly from GitHub

curl -L https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-osx-universal.zip -o duckdb.zip

unzip duckdb.zip

sudo mv duckdb /usr/local/bin/

```



### Step 2: Install Python Package

```bash

# Install DuckDB Python package

pip install duckdb



# Install additional packages we'll need

pip install pandas fastapi uvicorn pyarrow

```



### Step 3: Verify Installation

```bash

# Test CLI

duckdb --version



# Test Python

python -c "import duckdb; print(duckdb.__version__)"

```



## Basic DuckDB Operations



### Starting DuckDB

```bash

# Start DuckDB CLI with in-memory database

duckdb



# Start with persistent database file

duckdb my_database.db

```



### Basic SQL Commands

```sql

-- Create a table

CREATE TABLE users (

    id INTEGER PRIMARY KEY,

    name VARCHAR,

    email VARCHAR,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);



-- Insert data

INSERT INTO users (name, email) VALUES

    ('Alice Johnson', 'alice@example.com'),

    ('Bob Smith', 'bob@example.com'),

    ('Carol Davis', 'carol@example.com');



-- Query data

SELECT * FROM users;



-- Show all tables

SHOW TABLES;



-- Describe table structure

DESCRIBE users;

```



## Working with Different File Formats



### CSV Files

```sql

-- Read CSV file

SELECT * FROM 'data.csv';



-- Read CSV with options

SELECT * FROM read_csv_auto('data.csv', header=true, delimiter=',');



-- Create table from CSV

CREATE TABLE sales AS SELECT * FROM 'sales_data.csv';



-- Export to CSV

COPY (SELECT * FROM users) TO 'users_export.csv' WITH (FORMAT CSV, HEADER);

```



### Parquet Files

```sql

-- Read Parquet file

SELECT * FROM 'data.parquet';



-- Create table from Parquet

CREATE TABLE products AS SELECT * FROM 'products.parquet';



-- Export to Parquet

COPY (SELECT * FROM users) TO 'users_export.parquet' (FORMAT PARQUET);

```



### JSON Files

```sql

-- Read JSON file

SELECT * FROM 'data.json';



-- Read JSON with specific structure

SELECT * FROM read_json_auto('data.json');



-- Export to JSON

COPY (SELECT * FROM users) TO 'users_export.json' (FORMAT JSON);

```



### Excel Files

```sql

-- Install spatial extension for Excel support

INSTALL spatial;

LOAD spatial;



-- Read Excel file

SELECT * FROM 'data.xlsx';

```



## Persistent Storage



### Creating Persistent Database

```sql

-- Connect to persistent database file

.open persistent_db.duckdb



-- Create tables that will persist

CREATE TABLE employees (

    employee_id INTEGER PRIMARY KEY,

    first_name VARCHAR(50),

    last_name VARCHAR(50),

    department VARCHAR(50),

    salary DECIMAL(10,2),

    hire_date DATE

);



-- Insert data

INSERT INTO employees VALUES

    (1, 'John', 'Doe', 'Engineering', 75000.00, '2023-01-15'),

    (2, 'Jane', 'Smith', 'Marketing', 65000.00, '2023-02-20'),

    (3, 'Mike', 'Johnson', 'Sales', 55000.00, '2023-03-10');



-- Create indexes for better performance

CREATE INDEX idx_department ON employees(department);

CREATE INDEX idx_salary ON employees(salary);

```



### Database Backup and Restore

```sql

-- Export entire database

EXPORT DATABASE 'backup_directory';



-- Import database

IMPORT DATABASE 'backup_directory';

```



## Python Integration



### Basic Python Usage

```python

import duckdb

import pandas as pd



# Connect to in-memory database

conn = duckdb.connect()



# Connect to persistent database

# conn = duckdb.connect('my_database.db')



# Execute SQL

result = conn.execute("SELECT 42 as answer").fetchall()

print(result)



# Create table and insert data

conn.execute("""

    CREATE TABLE products (

        id INTEGER PRIMARY KEY,

        name VARCHAR,

        price DECIMAL(10,2),

        category VARCHAR

    )

""")



conn.execute("""

    INSERT INTO products VALUES

    (1, 'Laptop', 999.99, 'Electronics'),

    (2, 'Book', 19.99, 'Education'),

    (3, 'Coffee Mug', 12.99, 'Kitchen')

""")



# Query to pandas DataFrame

df = conn.execute("SELECT * FROM products").df()

print(df)

```



### Working with Pandas

```python

import pandas as pd

import duckdb



# Create sample DataFrame

df = pd.DataFrame({

    'id': [1, 2, 3, 4, 5],

    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],

    'score': [95, 87, 92, 88, 91],

    'grade': ['A', 'B', 'A', 'B', 'A']

})



# Connect to DuckDB

conn = duckdb.connect('school.db')



# Register DataFrame as a table

conn.register('students_df', df)



# Query the DataFrame using SQL

result = conn.execute("""

    SELECT grade, AVG(score) as avg_score, COUNT(*) as count

    FROM students_df

    GROUP BY grade

    ORDER BY avg_score DESC

""").df()



print(result)



# Create permanent table from DataFrame

conn.execute("CREATE TABLE students AS SELECT * FROM students_df")

```



### File Format Operations in Python

```python

import duckdb



conn = duckdb.connect('data_warehouse.db')



# Read different file formats

# CSV

conn.execute("CREATE TABLE sales AS SELECT * FROM 'sales_data.csv'")



# Parquet

conn.execute("CREATE TABLE inventory AS SELECT * FROM 'inventory.parquet'")



# JSON

conn.execute("CREATE TABLE logs AS SELECT * FROM 'application_logs.json'")



# Query across different sources

result = conn.execute("""

    SELECT 

        s.product_id,

        s.quantity_sold,

        i.stock_quantity,

        (i.stock_quantity - s.quantity_sold) as remaining_stock

    FROM sales s

    JOIN inventory i ON s.product_id = i.product_id

    WHERE s.sale_date >= '2024-01-01'

""").df()



# Export results

conn.execute("COPY (SELECT * FROM result) TO 'analysis_results.parquet' (FORMAT PARQUET)")

```



## FastAPI Integration



### Complete FastAPI Application

```python

from fastapi import FastAPI, HTTPException, Query

from pydantic import BaseModel

from typing import List, Optional

import duckdb

import pandas as pd

from datetime import datetime



app = FastAPI(title="DuckDB API", description="API for DuckDB operations")



# Database connection

DB_PATH = "fastapi_app.db"



def get_db_connection():

    return duckdb.connect(DB_PATH)



# Pydantic models

class User(BaseModel):

    name: str

    email: str

    age: Optional[int] = None



class UserResponse(BaseModel):

    id: int

    name: str

    email: str

    age: Optional[int]

    created_at: datetime



# Initialize database

@app.on_event("startup")

async def startup_event():

    conn = get_db_connection()

    conn.execute("""

        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY,

            name VARCHAR NOT NULL,

            email VARCHAR UNIQUE NOT NULL,

            age INTEGER,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )

    """)

    conn.close()



# API endpoints

@app.post("/users/", response_model=UserResponse)

async def create_user(user: User):

    conn = get_db_connection()

    try:

        result = conn.execute("""

            INSERT INTO users (name, email, age)

            VALUES (?, ?, ?)

            RETURNING *

        """, [user.name, user.email, user.age]).fetchone()

        

        return UserResponse(

            id=result[0],

            name=result[1],

            email=result[2],

            age=result[3],

            created_at=result[4]

        )

    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))

    finally:

        conn.close()



@app.get("/users/", response_model=List[UserResponse])

async def get_users(

    skip: int = Query(0, ge=0),

    limit: int = Query(100, ge=1, le=1000)

):

    conn = get_db_connection()

    try:

        results = conn.execute("""

            SELECT * FROM users

            ORDER BY created_at DESC

            LIMIT ? OFFSET ?

        """, [limit, skip]).fetchall()

        

        return [

            UserResponse(

                id=row[0],

                name=row[1],

                email=row[2],

                age=row[3],

                created_at=row[4]

            )

            for row in results

        ]

    finally:

        conn.close()



@app.get("/users/{user_id}", response_model=UserResponse)

async def get_user(user_id: int):

    conn = get_db_connection()

    try:

        result = conn.execute("""

            SELECT * FROM users WHERE id = ?

        """, [user_id]).fetchone()

        

        if not result:

            raise HTTPException(status_code=404, detail="User not found")

        

        return UserResponse(

            id=result[0],

            name=result[1],

            email=result[2],

            age=result[3],

            created_at=result[4]

        )

    finally:

        conn.close()



@app.post("/upload-csv/")

async def upload_csv(table_name: str):

    """Upload CSV data to DuckDB"""

    conn = get_db_connection()

    try:

        # Example: Load CSV file into table

        conn.execute(f"""

            CREATE OR REPLACE TABLE {table_name} AS 

            SELECT * FROM 'uploaded_data.csv'

        """)

        

        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        return {"message": f"Successfully loaded {count} rows into {table_name}"}

    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))

    finally:

        conn.close()



@app.get("/analytics/summary")

async def get_analytics_summary():

    """Get database analytics summary"""

    conn = get_db_connection()

    try:

        # Get table information

        tables = conn.execute("SHOW TABLES").fetchall()

        

        summary = {"tables": {}}

        for table in tables:

            table_name = table[0]

            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            summary["tables"][table_name] = {"row_count": count}

        

        return summary

    finally:

        conn.close()



# Run with: uvicorn main:app --reload

```



## MotherDuck Integration



### Setting up MotherDuck

```python

import duckdb

import os



# Set up MotherDuck token (get from https://motherduck.com)

# You can set this as an environment variable

os.environ['motherduck_token'] = 'your_motherduck_token_here'



# Or pass it directly in the connection string

def connect_to_motherduck():

    # Connect to MotherDuck

    conn = duckdb.connect('md:your_database_name')

    return conn



# Alternative connection with token

def connect_with_token(token: str, database: str):

    conn = duckdb.connect(f'md:{database}?motherduck_token={token}')

    return conn

```



### MotherDuck Operations

```python

import duckdb



# Connect to MotherDuck

conn = duckdb.connect('md:my_cloud_db')



# Create cloud table

conn.execute("""

    CREATE TABLE cloud_sales (

        sale_id INTEGER PRIMARY KEY,

        product_name VARCHAR,

        sale_amount DECIMAL(10,2),

        sale_date DATE,

        customer_id INTEGER

    )

""")



# Insert data

conn.execute("""

    INSERT INTO cloud_sales VALUES

    (1, 'Laptop Pro', 1299.99, '2024-01-15', 101),

    (2, 'Wireless Mouse', 29.99, '2024-01-16', 102),

    (3, 'Keyboard', 79.99, '2024-01-17', 103)

""")



# Sync local data to MotherDuck

local_conn = duckdb.connect('local_database.db')



# Copy from local to cloud

conn.execute("""

    CREATE TABLE cloud_users AS 

    SELECT * FROM 'local_database.db'.users

""")



# Query cloud data

cloud_data = conn.execute("""

    SELECT 

        DATE_TRUNC('month', sale_date) as month,

        SUM(sale_amount) as total_sales,

        COUNT(*) as transaction_count

    FROM cloud_sales

    GROUP BY DATE_TRUNC('month', sale_date)

    ORDER BY month

""").df()



print(cloud_data)

```



### Hybrid Local-Cloud Operations

```python

import duckdb



# Connect to both local and cloud

local_conn = duckdb.connect('local_warehouse.db')

cloud_conn = duckdb.connect('md:analytics_db')



# Create local staging table

local_conn.execute("""

    CREATE TABLE staging_orders AS

    SELECT * FROM 'daily_orders.csv'

""")



# Process and clean data locally

processed_data = local_conn.execute("""

    SELECT 

        order_id,

        customer_id,

        order_amount,

        order_date,

        CASE 

            WHEN order_amount > 1000 THEN 'High Value'

            WHEN order_amount > 100 THEN 'Medium Value'

            ELSE 'Low Value'

        END as order_category

    FROM staging_orders

    WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'

""").df()



# Upload to MotherDuck

cloud_conn.register('processed_orders', processed_data)

cloud_conn.execute("""

    CREATE OR REPLACE TABLE orders_analytics AS

    SELECT * FROM processed_orders

""")



# Run analytics in the cloud

analytics_result = cloud_conn.execute("""

    SELECT 

        order_category,

        COUNT(*) as order_count,

        AVG(order_amount) as avg_order_value,

        SUM(order_amount) as total_revenue

    FROM orders_analytics

    GROUP BY order_category

    ORDER BY total_revenue DESC

""").df()



print(analytics_result)

```



## Complete Example Project



### Project Structure

```

duckdb_project/

├── main.py              # FastAPI application

├── database.py          # Database utilities

├── models.py            # Pydantic models

├── config.py            # Configuration

├── data/

│   ├── sample.csv

│   ├── products.parquet

│   └── logs.json

└── requirements.txt

```



### requirements.txt

```

duckdb>=0.9.0

fastapi>=0.104.0

uvicorn>=0.24.0

pandas>=2.0.0

pyarrow>=14.0.0

pydantic>=2.0.0

python-multipart>=0.0.6

```



### Complete Application (main.py)

```python

from fastapi import FastAPI, HTTPException, UploadFile, File

from fastapi.responses import FileResponse

import duckdb

import pandas as pd

import os

from datetime import datetime, date

import tempfile



app = FastAPI(

    title="Advanced DuckDB API",

    description="Complete DuckDB integration with file handling and analytics"

)



# Global database connection

DB_PATH = "production.db"



class DatabaseManager:

    def __init__(self, db_path: str):

        self.db_path = db_path

        self.init_database()

    

    def get_connection(self):

        return duckdb.connect(self.db_path)

    

    def init_database(self):

        conn = self.get_connection()

        

        # Create core tables

        conn.execute("""

            CREATE TABLE IF NOT EXISTS products (

                id INTEGER PRIMARY KEY,

                name VARCHAR NOT NULL,

                price DECIMAL(10,2),

                category VARCHAR,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

            )

        """)

        

        conn.execute("""

            CREATE TABLE IF NOT EXISTS sales (

                id INTEGER PRIMARY KEY,

                product_id INTEGER,

                quantity INTEGER,

                sale_date DATE,

                customer_id INTEGER,

                FOREIGN KEY (product_id) REFERENCES products(id)

            )

        """)

        

        conn.close()



db_manager = DatabaseManager(DB_PATH)



@app.post("/upload/csv")

async def upload_csv(file: UploadFile = File(...), table_name: str = "uploaded_data"):

    """Upload and process CSV file"""

    if not file.filename.endswith('.csv'):

        raise HTTPException(status_code=400, detail="File must be CSV format")

    

    # Save uploaded file temporarily

    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:

        content = await file.read()

        tmp_file.write(content)

        tmp_file_path = tmp_file.name

    

    try:

        conn = db_manager.get_connection()

        

        # Load CSV into DuckDB

        conn.execute(f"""

            CREATE OR REPLACE TABLE {table_name} AS

            SELECT * FROM read_csv_auto('{tmp_file_path}')

        """)

        

        # Get row count

        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        

        # Get column info

        columns = conn.execute(f"DESCRIBE {table_name}").fetchall()

        

        conn.close()

        

        return {

            "message": f"Successfully uploaded {count} rows",

            "table_name": table_name,

            "columns": [{"name": col[0], "type": col[1]} for col in columns]

        }

    

    finally:

        os.unlink(tmp_file_path)



@app.get("/export/{table_name}")

async def export_table(table_name: str, format: str = "csv"):

    """Export table to various formats"""

    conn = db_manager.get_connection()

    

    try:

        # Check if table exists

        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

        if table_name not in tables:

            raise HTTPException(status_code=404, detail="Table not found")

        

        # Create temporary file

        if format.lower() == "csv":

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')

            conn.execute(f"""

                COPY (SELECT * FROM {table_name}) 

                TO '{temp_file.name}' 

                WITH (FORMAT CSV, HEADER)

            """)

            media_type = 'text/csv'

        

        elif format.lower() == "parquet":

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')

            conn.execute(f"""

                COPY (SELECT * FROM {table_name}) 

                TO '{temp_file.name}' 

                (FORMAT PARQUET)

            """)

            media_type = 'application/octet-stream'

        

        else:

            raise HTTPException(status_code=400, detail="Unsupported format")

        

        temp_file.close()

        

        return FileResponse(

            path=temp_file.name,

            filename=f"{table_name}.{format}",

            media_type=media_type

        )

    

    finally:

        conn.close()



@app.get("/analytics/dashboard")

async def get_dashboard_data():

    """Get comprehensive analytics dashboard data"""

    conn = db_manager.get_connection()

    

    try:

        # Sales by month

        monthly_sales = conn.execute("""

            SELECT 

                DATE_TRUNC('month', sale_date) as month,

                SUM(p.price * s.quantity) as revenue,

                COUNT(*) as transaction_count

            FROM sales s

            JOIN products p ON s.product_id = p.id

            GROUP BY DATE_TRUNC('month', sale_date)

            ORDER BY month

        """).df().to_dict('records')

        

        # Top products

        top_products = conn.execute("""

            SELECT 

                p.name,

                SUM(s.quantity) as total_sold,

                SUM(p.price * s.quantity) as revenue

            FROM sales s

            JOIN products p ON s.product_id = p.id

            GROUP BY p.id, p.name

            ORDER BY revenue DESC

            LIMIT 10

        """).df().to_dict('records')

        

        # Category performance

        category_performance = conn.execute("""

            SELECT 

                p.category,

                COUNT(DISTINCT p.id) as product_count,

                SUM(s.quantity) as units_sold,

                SUM(p.price * s.quantity) as revenue

            FROM products p

            LEFT JOIN sales s ON p.id = s.product_id

            GROUP BY p.category

            ORDER BY revenue DESC

        """).df().to_dict('records')

        

        return {

            "monthly_sales": monthly_sales,

            "top_products": top_products,

            "category_performance": category_performance,

            "generated_at": datetime.now().isoformat()

        }

    

    finally:

        conn.close()



# Run the application

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

```



### Running the Complete Application



1. **Install dependencies:**

```bash

pip install -r requirements.txt

```



2. **Run the FastAPI server:**

```bash

uvicorn main:app --reload

```



3. **Access the API:**

- API Documentation: http://localhost:8000/docs

- Interactive API: http://localhost:8000/redoc



4. **Connect to MotherDuck:**

```python

# In your Python scripts

import duckdb

conn = duckdb.connect('md:your_database?motherduck_token=your_token')

```



This tutorial covers everything from basic installation to advanced cloud integration. You now have a complete toolkit for working with DuckDB in various scenarios!
