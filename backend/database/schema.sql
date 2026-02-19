CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    income REAL,
    loan REAL,
    emi REAL,
    tenure REAL,
    ontime REAL,
    delays REAL,
    credit REAL,
    risk TEXT,
    probability REAL,
    input_data TEXT,
    batch_upload_id INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(batch_upload_id) REFERENCES csv_uploads(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS emi_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    emi_amount REAL,
    due_date TEXT,
    paid_date TEXT,
    status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS csv_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    filename TEXT,
    file_type TEXT,
    total_records INTEGER,
    high_risk_count INTEGER,
    low_risk_count INTEGER,
    pdf_filename TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS batch_prediction_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    batch_upload_id INTEGER NOT NULL,
    row_index INTEGER,
    income REAL,
    loan REAL,
    emi REAL,
    tenure REAL,
    ontime REAL,
    delays REAL,
    credit REAL,
    risk TEXT,
    probability REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(batch_upload_id) REFERENCES csv_uploads(id) ON DELETE CASCADE
);
