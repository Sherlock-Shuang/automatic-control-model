import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
         raise ValueError("Missing Supabase credentials in .env")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def similarity_search(query_embedding: list[float], top_k: int = 3):
    """
    Perform a vector similarity search via Supabase RPC.
    Assumes an RPC function `match_textbook_knowledge` exists in Postgres.
    """
    client = get_supabase_client()
    # this relies on a supabase pgvector RPC match function
    response = client.rpc('match_textbook_knowledge', {
        'query_embedding': query_embedding,
        'match_threshold': 0.5,
        'match_count': top_k
    }).execute()
    return response.data
