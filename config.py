"""Environment-backed configuration; no account or access key is bundled."""
import os
from pathlib import Path

path = str(Path(os.environ.get('MODEL_ARTIFACT_DIR', 'artifacts/models')).resolve())
top = int(os.environ.get('RECOMMENDATION_TOP_K', '10'))
model_key = os.environ.get('MODEL_API_KEY', '')
interactions_api_url = os.environ.get('INTERACTIONS_API_URL', '')
api_page_size = int(os.environ.get('INTERACTIONS_API_PAGE_SIZE', '1000'))
mongo_client_website = os.environ.get('INTERACTIONS_MONGO_URI', '')
