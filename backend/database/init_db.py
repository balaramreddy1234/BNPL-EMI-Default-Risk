from database.db_connection import get_db

def init_database():
    db = get_db()
    try:
        # 1. Users Table
        # Includes Name, Email, Password, and new profile fields (Age, Address)
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, 
                email TEXT UNIQUE NOT NULL, 
                password TEXT NOT NULL,
                age INTEGER,      -- Field for User Profile
                address TEXT,     -- Field for User Profile
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. Unified Predictions Table
        # Stores both 'Manual' risk checks and 'Batch' CSV upload summaries
        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                source TEXT NOT NULL,           -- 'manual' or 'batch'
                
                -- Manual Entry Fields
                income REAL, 
                loan REAL, 
                emi REAL, 
                tenure REAL,
                ontime REAL, 
                delays REAL, 
                credit REAL,
                risk TEXT, 
                probability REAL,
                report_path TEXT,               -- For manual PDF reports
                
                -- Batch Entry Fields
                filename TEXT,                  -- Original CSV name
                high_risk_count INTEGER,
                low_risk_count INTEGER,
                total_count INTEGER,            -- Total rows in CSV
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        
        db.commit()
        print("✅ Database fully initialized: User profiles and Prediction history ready.")
        
    except Exception as e:
        print(f"❌ Database Init Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    init_database()