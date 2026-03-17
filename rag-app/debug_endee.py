from endee import Endee, Precision
import os
from dotenv import load_dotenv

load_dotenv()
ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:8080")

client = Endee()
client.set_base_url(f"{ENDEE_HOST}/api/v1")

try:
    # Ensure index exists
    existing = [idx.name for idx in client.list_indexes()] if hasattr(client.list_indexes(), "__iter__") else []
    print(f"Initial existing: {existing}")
    
    if "test_index" not in existing:
        client.create_index(name="test_index", dimension=384, space_type="cosine", precision=Precision.INT8)
        print("Created test_index")
    
    indexes_resp = client.list_indexes()
    print(f"Full indexes response: {indexes_resp}")
    print(f"Type of response: {type(indexes_resp)}")
    
    for idx in indexes_resp:
        print(f"Index item: {idx}")
        print(f"Index item type: {type(idx)}")
        print(f"Index name: {idx.name}")
        
except Exception as e:
    import traceback
    traceback.print_exc()
