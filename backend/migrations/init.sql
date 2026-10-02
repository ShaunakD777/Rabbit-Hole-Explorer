-- Enable pgvector extension (must run before SQLAlchemy creates tables)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
