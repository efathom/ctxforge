-- CtxForge PostgreSQL Initialization
-- This script creates the necessary tables for the demo
-- Schema must match what postgres_store.py expects

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    state JSONB NOT NULL DEFAULT '{}',
    data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);

-- Memories table (must match postgres_store.py schema)
CREATE TABLE IF NOT EXISTS memories (
    memory_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL DEFAULT 0.5,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    tags TEXT[] NOT NULL DEFAULT '{}',
    data JSONB NOT NULL DEFAULT '{}',
    embedding FLOAT[],
    headline VARCHAR(150),
    subtitle VARCHAR(300),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    search_vector TSVECTOR
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_search ON memories USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active);
CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at);

-- Function to update search vector
CREATE OR REPLACE FUNCTION update_memory_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.content, ''));
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for search vector
DROP TRIGGER IF EXISTS memory_search_vector_update ON memories;
CREATE TRIGGER memory_search_vector_update
    BEFORE INSERT OR UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_search_vector();

-- =====================================
-- Expertise Tables
-- =====================================

-- Expertise table (knowledge bases)
CREATE TABLE IF NOT EXISTS expertise (
    expertise_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255),
    version INTEGER NOT NULL DEFAULT 1,
    token_budget INTEGER NOT NULL DEFAULT 80000,
    next_item_id INTEGER NOT NULL DEFAULT 1,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_expertise_domain ON expertise(domain);
CREATE INDEX IF NOT EXISTS idx_expertise_updated_at ON expertise(updated_at);

-- Expertise items table
CREATE TABLE IF NOT EXISTS expertise_items (
    item_id VARCHAR(255) PRIMARY KEY,
    expertise_id VARCHAR(255) NOT NULL REFERENCES expertise(expertise_id) ON DELETE CASCADE,
    section VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    helpful_count INTEGER NOT NULL DEFAULT 0,
    harmful_count INTEGER NOT NULL DEFAULT 0,
    source VARCHAR(100),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    embedding FLOAT[],
    search_vector TSVECTOR,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_expertise_items_expertise_id ON expertise_items(expertise_id);
CREATE INDEX IF NOT EXISTS idx_expertise_items_section ON expertise_items(section);
CREATE INDEX IF NOT EXISTS idx_expertise_items_active ON expertise_items(expertise_id, is_active);
CREATE INDEX IF NOT EXISTS idx_expertise_items_search ON expertise_items USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_expertise_items_effectiveness ON expertise_items(helpful_count, harmful_count);

-- Expertise usage logs table
CREATE TABLE IF NOT EXISTS expertise_usage_logs (
    log_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    expertise_id VARCHAR(255) NOT NULL REFERENCES expertise(expertise_id) ON DELETE CASCADE,
    items_used TEXT[] NOT NULL DEFAULT '{}',
    feedback JSONB NOT NULL DEFAULT '{}',
    outcome VARCHAR(50),
    context_summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_expertise_usage_session ON expertise_usage_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_expertise_usage_expertise ON expertise_usage_logs(expertise_id);
CREATE INDEX IF NOT EXISTS idx_expertise_usage_created ON expertise_usage_logs(created_at);

-- Function to update expertise item search vector
CREATE OR REPLACE FUNCTION update_expertise_item_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.content, ''));
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for expertise item search vector
DROP TRIGGER IF EXISTS expertise_item_search_vector_update ON expertise_items;
CREATE TRIGGER expertise_item_search_vector_update
    BEFORE INSERT OR UPDATE ON expertise_items
    FOR EACH ROW
    EXECUTE FUNCTION update_expertise_item_search_vector();

-- Grant permissions (run as superuser if needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO contextengine;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO contextengine;
