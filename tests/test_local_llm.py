import json
import os
import tempfile
import unittest
from unittest.mock import patch

_runtime = tempfile.TemporaryDirectory()
os.environ['MUA_DATA_DIR'] = _runtime.name
from src import local_llm, llm_engine
from app import app
from fastapi.testclient import TestClient


class LocalInferenceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_loopback_only(self):
        for address in ('https://example.com', 'http://192.168.1.1:8000', 'http://localhost@evil.test', 'file:///tmp/model', 'http://localhost:11434/?x=1'):
            with self.assertRaises(ValueError):
                local_llm.validate_url(address)
        self.assertEqual(local_llm.validate_url('http://127.0.0.1:1234/v1/'), 'http://127.0.0.1:1234/v1')

    @patch('src.local_llm.request_json')
    def test_ollama_request_and_response(self, request):
        request.return_value = {'message': {'content': 'Hello Mohit'}}
        self.assertEqual(local_llm.chat('ollama', 'http://localhost:11434', 'test-model', [{'role':'user','content':'hello'}]), 'Hello Mohit')
        args = request.call_args.args
        self.assertEqual(args[1], '/api/chat')
        self.assertFalse(args[2]['stream'])
        self.assertEqual(args[2]['model'], 'test-model')

    @patch('src.local_llm.request_json')
    def test_compatible_runtime(self, request):
        request.return_value = {'choices':[{'message':{'content':'Local answer'}}]}
        self.assertEqual(local_llm.chat('local-api','http://localhost:1234/v1','loaded-model',[]),'Local answer')
        self.assertEqual(request.call_args.args[1], '/chat/completions')
        request.return_value = {'data':[{'id':'loaded-model'}]}
        self.assertEqual(local_llm.discover('local-api','http://localhost:1234/v1')[0]['name'],'loaded-model')

    @patch('src.llm_engine.local_chat', return_value='Personal answer')
    def test_history_and_profile_context(self, chat):
        response = self.client.post('/llm/chat',json={'name':'mohit','message':'And my music?','mode':'ollama','history':[{'role':'user','content':'What is my schedule?'},{'role':'assistant','content':'Your math class.'}]})
        self.assertEqual(response.status_code,200)
        messages = chat.call_args.args[3]
        self.assertEqual(messages[1]['content'],'What is my schedule?')
        self.assertIn('Mohit',messages[-1]['content'])
        self.assertEqual(response.json()['error'],None)

    @patch('src.llm_engine.local_chat', side_effect=RuntimeError('Runtime offline'))
    def test_unavailable_runtime_does_not_call_cloud(self, chat):
        with patch('src.llm_engine._generate_gemini') as cloud:
            result = llm_engine.generate('Mohit','profile','hello','chat',mode='ollama')
        self.assertEqual(result['error'],'Runtime offline')
        cloud.assert_not_called()

    def test_validation_and_workspace_creation(self):
        self.assertEqual(self.client.post('/llm/chat',json={'name':'mohit','message':'','mode':'ollama'}).status_code,422)
        self.assertEqual(self.client.post('/llm/chat',json={'name':'mohit','message':'hi','mode':'remote'}).status_code,422)
        self.assertEqual(self.client.post('/llm/config',json={'ollama_url':'http://evil.test'}).status_code,400)
        self.assertEqual(self.client.post('/profiles/mohit/create',json={}).status_code,409)
        response=self.client.post('/profiles/test-person/create',json={'display_name':'Test Person'})
        self.assertEqual(response.status_code,200)
        workspace=response.json()['profile']['workspace']
        self.assertEqual(workspace['music']['playlists'],[])
        self.assertEqual(workspace['fitness']['steps_today'],0)
        self.assertEqual(self.client.post('/profiles/test-person/create',json={}).status_code,409)
        self.assertNotIn('gemini_key',self.client.get('/llm/status').json())


if __name__=='__main__':
    unittest.main()
