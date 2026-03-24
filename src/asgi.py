"""src/asgi.py fallback entrypoint for Vercel."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT_APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
spec = spec_from_file_location("root_flask_app", ROOT_APP_PATH)
module = module_from_spec(spec)
spec.loader.exec_module(module)
app = module.app
