import unittest
from unittest.mock import patch
import app

class HttpBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_health_is_available_without_model_or_credentials(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'ok')

    def test_missing_key_disables_operations(self):
        with patch.object(app.config, 'model_key', ''):
            self.assertEqual(self.client.post('/', json={'act':'fit'}).status_code, 503)

    def test_authorized_invalid_action_and_untrained_model(self):
        with patch.object(app.config, 'model_key', 'offline-test-value'):
            self.assertEqual(self.client.post('/', json={'act':'recom'}).status_code, 401)
            headers = {'Authorization':'Bearer offline-test-value'}
            self.assertEqual(self.client.post('/', headers=headers, json={'act':'invalid'}).status_code, 400)
            with patch.object(app.core, 'model', None):
                self.assertEqual(self.client.post('/', headers=headers, json={'act':'recom'}).status_code, 409)
