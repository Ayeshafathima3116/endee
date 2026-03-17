import os
from endee import Endee
from dotenv import load_dotenv

load_dotenv()
host = os.getenv("ENDEE_HOST", "http://localhost:8080")
if not host.endswith("/api/v1"):
    host = host.rstrip("/") + "/api/v1"

client = Endee()
client.base_url = host

try:
    resp = client.list_indexes()
    print(f"List Indexes: {resp}")
    index = client.Index("documents")
    print(f"Describe documents: {index.describe()}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
