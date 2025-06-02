import sys
from pathlib import Path

# Add project root to PYTHONPATH to ensure imports work during tests
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root)) 