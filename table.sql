CREATE TABLE docs (
    id TEXT PRIMARY KEY,
    content TEXT,
    embedding JSONB,  -- or use ARRAY type
    included_in_qa BOOLEAN DEFAULT FALSE
);