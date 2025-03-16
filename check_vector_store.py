"""
Simple check to verify the TechDocVectorStore implementation after migration.
This script prints output directly to the console to avoid file writing issues.
"""

import os
import sys
from app.chat.vector_stores import get_vector_store

def main():
    """Print basic vector store information to verify migration success."""
    print("\n===== VECTOR STORE MIGRATION VERIFICATION =====")
    
    try:
        # Get the vector store instance
        vs = get_vector_store()
        
        # Check what type it is - should be TechDocVectorStore, not from storage.py
        print(f"Vector store type: {type(vs).__name__}")
        print(f"Initialized: {vs.initialized}")
        
        # Check if it has the new methods from vector_store.py
        methods = [method for method in dir(vs) if not method.startswith('_')]
        print(f"\nAvailable methods:")
        for method in sorted(methods):
            print(f"  - {method}")
        
        # Check for vector_store specific attributes
        if hasattr(vs, 'index_name'):
            print(f"\nUsing Pinecone index: {vs.index_name}")
        
        if hasattr(vs, 'metrics'):
            print(f"\nMetrics tracking enabled: {vs.metrics is not None}")
        
        # Verify storage.py is gone
        legacy_path = os.path.join('app', 'chat', 'vector_stores', 'storage.py')
        full_path = os.path.join(os.getcwd(), legacy_path)
        if os.path.exists(full_path):
            print(f"\n⚠️ WARNING: Legacy storage.py file still exists at {full_path}")
        else:
            print(f"\n✅ Legacy storage.py has been completely removed")
        
        # Check the available functions in the vector_stores module
        import app.chat.vector_stores
        module_items = dir(app.chat.vector_stores)
        print(f"\nVector stores module exports:")
        for item in sorted(module_items):
            if not item.startswith('_'):
                print(f"  - {item}")
        
        print("\n===== MIGRATION STATUS =====")
        if vs.__class__.__name__ == "TechDocVectorStore" and not os.path.exists(full_path):
            print("✅ MIGRATION SUCCESSFUL: The vector_store.py implementation is working")
            print("   and the legacy storage.py has been completely removed.")
        else:
            print("⚠️ MIGRATION INCOMPLETE: Please check the details above.")
            
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
