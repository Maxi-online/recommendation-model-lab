"""Lazy dispatch for the seven optional historical model adapters."""
from importlib import import_module
import config

ADAPTERS = [('ModelTfrs', 'ModelTfrs'), ('ModelFastai', 'ModelFastai'),
            ('ModelLightFM', 'ModelLightFM'), ('ModelGMF', 'ModelGMF'),
            ('ModelNCF', 'ModelNCF'), ('ModelCollaborativeF', 'ModelCollaborativeF'), ('NNHS', 'NNHS')]

def fit(core):
    if not config.interactions_api_url:
        raise ValueError('Set INTERACTIONS_API_URL before running a source-backed training adapter')
    index = int(core.req.get('model', 1)) - 1
    days = int(core.req.get('days', 30))
    if not 0 <= index < len(ADAPTERS) or days < 1:
        raise ValueError('Invalid adapter index or time window')
    module_name, class_name = ADAPTERS[index]
    adapter = getattr(import_module('classes.' + module_name), class_name)
    core.model = adapter(config)
    return core.model.fit(days)
