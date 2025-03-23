CREATE TABLE docs (
    id TEXT PRIMARY KEY,
    content TEXT,
    embedding JSONB, 
    included_in_qa BOOLEAN DEFAULT FALSE
);
