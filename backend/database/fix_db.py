import sqlite3
import os

# Use __file__ to get the path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to the 'database' folder in the same directory as this script
DB_PATH = os.path.join(BASE_DIR, "database", "bnpl.db") 

def upgrade_db():
    # Check if database exists before trying to connect
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database file not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print(f"🚀 Starting Database Upgrade on: {DB_PATH}")
    
    # List of new columns to add to the 'predictions' table
    new_columns = [
        ("filename", "TEXT"),
        ("high_risk_count", "INTEGER"),
        ("low_risk_count", "INTEGER"),
        ("total_count", "INTEGER")
    ]
    
    for col_name, col_type in new_columns:
        try:
            # SQL to add a column if it doesn't exist
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            print(f"✅ Added column: {col_name}")
        except sqlite3.OperationalError as e:
            # Handle the case where the column is already there
            if "duplicate column name" in str(e).lower():
                print(f"ℹ️ Column '{col_name}' already exists, skipping.")
            else:
                print(f"❌ Operational Error: {e}")
            
    conn.commit()
    conn.close()
    print("🏁 Database Upgrade Complete!")

if __name__ == "__main__":
    upgrade_db()