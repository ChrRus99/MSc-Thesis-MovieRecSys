# Alias the package as CRM
import sys
from pathlib import Path

module_path = Path(__file__).parent / "movie_recommendation_system"
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))

#sys.modules['CRM'] = sys.modules[__name__]