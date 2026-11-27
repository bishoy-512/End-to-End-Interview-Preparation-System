import sys
from pathlib import Path

# The job_analysis code uses flat imports (extraction., analysis., ...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
