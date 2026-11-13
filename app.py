"""Loopback HTTP adapter. No model, source account or key is initialized at import."""
import hmac
import os
import threading
from flask import Flask, jsonify, request
import config
from classes.Core import Core
import fit
import recom
import popular

app = Flask(__name__)
core = Core()
core.config = config
operation_lock = threading.Lock()

@app.get('/health')
def health():
    return jsonify(status='ok', model_loaded=core.model is not None)

@app.post('/')
def operate():
    if not config.model_key:
        return jsonify(error='MODEL_API_KEY is not configured'), 503
    expected = 'Bearer ' + config.model_key
    if not hmac.compare_digest(request.headers.get('Authorization', ''), expected):
        return jsonify(error='Unauthorized'), 401
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or body.get('act') not in {'fit', 'recom', 'popular'}:
        return jsonify(error='Provide an action: fit, recom or popular'), 400
    with operation_lock:
        core.req = body
        if body['act'] != 'fit' and core.model is None:
            return jsonify(error='No model has been trained in this process'), 409
        try:
            result = {'fit': fit.fit, 'recom': recom.recom, 'popular': popular.popular}[body['act']](core)
        except (ValueError, TypeError):
            return jsonify(error='Invalid request or missing runtime configuration'), 400
        except ImportError:
            return jsonify(error='Install the selected optional adapter dependencies'), 503
        return result

if __name__ == '__main__':
    app.run(host=os.environ.get('MODEL_API_HOST', '127.0.0.1'), port=int(os.environ.get('MODEL_API_PORT', '5000')))
