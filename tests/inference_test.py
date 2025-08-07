# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock
from absl.testing import absltest
from langextract import inference


class TestOllamaLanguageModel(absltest.TestCase):

  @mock.patch.object(inference.OllamaLanguageModel, "_ollama_query")
  def test_ollama_infer(self, mock_ollama_query):

    # Actuall full gemma2 response using Ollama.
    gemma_response = {
        "model": "gemma2:latest",
        "created_at": "2025-01-23T22:37:08.579440841Z",
        "response": "{'bus' : '**autóbusz**'} \n\n\n  \n",
        "done": True,
        "done_reason": "stop",
        "context": [
            106,
            1645,
            108,
            1841,
            603,
            1986,
            575,
            59672,
            235336,
            107,
            108,
            106,
            2516,
            108,
            9766,
            6710,
            235281,
            865,
            664,
            688,
            7958,
            235360,
            6710,
            235306,
            688,
            12990,
            235248,
            110,
            139,
            108,
        ],
        "total_duration": 24038204381,
        "load_duration": 21551375738,
        "prompt_eval_count": 15,
        "prompt_eval_duration": 633000000,
        "eval_count": 17,
        "eval_duration": 1848000000,
    }
    mock_ollama_query.return_value = gemma_response
    model = inference.OllamaLanguageModel(
        model="gemma2:latest",
        model_url="http://localhost:11434",
        structured_output_format="json",
    )
    batch_prompts = ["What is bus in Hungarian?"]
    results = list(model.infer(batch_prompts))

    mock_ollama_query.assert_called_once_with(
        prompt="What is bus in Hungarian?",
        model="gemma2:latest",
        structured_output_format="json",
        model_url="http://localhost:11434",
    )
    expected_results = [[
        inference.ScoredOutput(
            score=1.0, output="{'bus' : '**autóbusz**'} \n\n\n  \n"
        )
    ]]
    self.assertEqual(results, expected_results)


class TestClaudeLanguageModel(absltest.TestCase):

  def test_claude_seed_parameter_initialization(self):
    """Test that ClaudeLanguageModel properly initializes with seed parameter."""
    test_seed = 12345
    
    # This will fail initialization due to no API key, but we can catch that
    # and verify the seed parameter is properly handled
    try:
      model = inference.ClaudeLanguageModel(
          api_key="test_key",
          seed=test_seed
      )
      self.assertEqual(model.seed, test_seed)
    except ValueError as e:
      # Expected error due to invalid API key
      if "API key not provided" in str(e):
        # Test that the parameter is accepted in the constructor
        pass
      else:
        raise

  @mock.patch('anthropic.Anthropic')
  def test_claude_seed_stored_but_not_passed(self, mock_anthropic):
    """Test that seed parameter is stored but not passed to Claude API (unsupported)."""
    mock_client = mock.Mock()
    mock_anthropic.return_value = mock_client
    
    # Mock the API response
    mock_response = mock.Mock()
    mock_response.content = [mock.Mock(text="Test response")]
    mock_client.messages.create.return_value = mock_response
    
    test_seed = 42
    model = inference.ClaudeLanguageModel(
        api_key="test_key",
        seed=test_seed
    )
    
    # Test single prompt processing
    result = model._process_single_prompt("Test prompt", {})
    
    # Verify the API was called WITHOUT seed parameter (since it's not supported)
    mock_client.messages.create.assert_called_once()
    call_args = mock_client.messages.create.call_args
    api_params = call_args[1] if call_args[1] else call_args[0][0]
    
    # Seed should NOT be in the API call since it's not supported
    self.assertNotIn('extra_body', api_params)
    self.assertNotIn('seed', api_params)
    
    # But the model should still store the seed value
    self.assertEqual(model.seed, test_seed)

  @mock.patch('anthropic.Anthropic')
  def test_claude_seed_from_kwargs_not_passed(self, mock_anthropic):
    """Test that seed parameter from kwargs is processed but not passed to API."""
    mock_client = mock.Mock()
    mock_anthropic.return_value = mock_client
    
    # Mock the API response
    mock_response = mock.Mock()
    mock_response.content = [mock.Mock(text="Test response")]
    mock_client.messages.create.return_value = mock_response
    
    model = inference.ClaudeLanguageModel(api_key="test_key")
    
    # Test with seed passed via kwargs
    runtime_seed = 999
    list(model.infer(["Test prompt"], seed=runtime_seed))
    
    # Verify the API was called WITHOUT seed parameter (since it's not supported)
    mock_client.messages.create.assert_called_once()
    call_args = mock_client.messages.create.call_args
    api_params = call_args[1] if call_args[1] else call_args[0][0]
    
    # Seed should NOT be in the API call since it's not supported
    self.assertNotIn('extra_body', api_params)
    self.assertNotIn('seed', api_params)


class TestHFLanguageModel(absltest.TestCase):

  def test_hf_model_initialization_with_api_key(self):
    """Test that HFLanguageModel properly initializes with API key."""
    test_api_key = "test_hf_token"
    test_model = "openai/gpt-oss-120b:cerebras"
    
    with mock.patch('openai.OpenAI') as mock_openai:
      model = inference.HFLanguageModel(
          api_key=test_api_key,
          model_id=test_model
      )
      
      self.assertEqual(model.api_key, test_api_key)
      self.assertEqual(model.model_id, test_model)
      self.assertEqual(model.base_url, 'https://router.huggingface.co/v1')
      
      # Verify OpenAI client was initialized with correct parameters
      mock_openai.assert_called_once_with(
          base_url='https://router.huggingface.co/v1',
          api_key=test_api_key
      )

  @mock.patch.dict('os.environ', {'HF_TOKEN': 'env_token'})
  def test_hf_model_initialization_with_env_token(self):
    """Test that HFLanguageModel uses HF_TOKEN from environment."""
    with mock.patch('openai.OpenAI') as mock_openai:
      model = inference.HFLanguageModel()
      
      self.assertEqual(model.api_key, 'env_token')
      mock_openai.assert_called_once_with(
          base_url='https://router.huggingface.co/v1',
          api_key='env_token'
      )

  def test_hf_model_initialization_without_token_raises_error(self):
    """Test that HFLanguageModel raises error when no token is available."""
    with mock.patch.dict('os.environ', {}, clear=True):
      with self.assertRaises(ValueError) as cm:
        inference.HFLanguageModel()
      
      self.assertIn('HF_TOKEN not provided', str(cm.exception))

  @mock.patch('openai.OpenAI')
  def test_hf_single_prompt_processing(self, mock_openai):
    """Test single prompt processing with HFLanguageModel."""
    mock_client = mock.Mock()
    mock_openai.return_value = mock_client
    
    # Mock the API response
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = "Test response from HuggingFace"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    from langextract import data
    model = inference.HFLanguageModel(
        api_key="test_token",
        format_type=data.FormatType.YAML  # Use YAML to avoid JSON prompt modification
    )
    
    result = model._process_single_prompt("Test prompt", {})
    
    # Verify the API call
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    api_params = call_args[1] if call_args[1] else call_args[0]
    
    expected_params = {
        'model': 'openai/gpt-oss-120b:cerebras',
        'max_tokens': 1024,
        'temperature': 0.0,
        'messages': [{'role': 'user', 'content': 'Test prompt'}]
    }
    
    for key, value in expected_params.items():
      self.assertEqual(api_params[key], value)
    
    # Verify the result
    expected_result = inference.ScoredOutput(
        score=1.0, 
        output="Test response from HuggingFace"
    )
    self.assertEqual(result, expected_result)

  @mock.patch('openai.OpenAI')
  def test_hf_infer_with_batch_prompts(self, mock_openai):
    """Test inference with multiple prompts."""
    mock_client = mock.Mock()
    mock_openai.return_value = mock_client
    
    # Mock the API response
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = "Batch response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    model = inference.HFLanguageModel(api_key="test_token", max_workers=1)
    
    batch_prompts = ["Prompt 1", "Prompt 2"]
    results = list(model.infer(batch_prompts))
    
    # Should have been called twice (once per prompt)
    self.assertEqual(mock_client.chat.completions.create.call_count, 2)
    
    # Should return two results
    self.assertEqual(len(results), 2)
    expected_result = [inference.ScoredOutput(score=1.0, output="Batch response")]
    for result in results:
      self.assertEqual(result, expected_result)

  @mock.patch('openai.OpenAI')
  def test_hf_structured_output(self, mock_openai):
    """Test structured output handling."""
    mock_client = mock.Mock()
    mock_openai.return_value = mock_client
    
    # Mock the API response
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = '{"key": "value"}'
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    # Create a mock schema
    mock_schema = mock.Mock()
    mock_schema.openai_schema = {"type": "object", "properties": {"key": {"type": "string"}}}
    
    model = inference.HFLanguageModel(
        api_key="test_token",
        structured_schema=mock_schema
    )
    
    model._process_single_prompt("Test prompt", {})
    
    # Verify structured output was requested via prompt instruction (not response_format)
    call_args = mock_client.chat.completions.create.call_args
    api_params = call_args[1] if call_args[1] else call_args[0]
    
    # HuggingFace doesn't use response_format, so it shouldn't be present
    self.assertNotIn('response_format', api_params)
    
    # Instead, check that the prompt was modified to include schema instructions
    prompt_content = api_params['messages'][0]['content']
    self.assertIn('Please respond in valid JSON format matching this schema', prompt_content)
    self.assertIn('type', prompt_content)
    self.assertIn('object', prompt_content)

  @mock.patch('openai.OpenAI')
  def test_hf_json_format_type(self, mock_openai):
    """Test JSON format type handling."""
    mock_client = mock.Mock()
    mock_openai.return_value = mock_client
    
    # Mock the API response
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = '{"result": "json"}'
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    from langextract import data
    model = inference.HFLanguageModel(
        api_key="test_token",
        format_type=data.FormatType.JSON
    )
    
    model._process_single_prompt("Test prompt", {})
    
    # Verify JSON format was requested via prompt instruction (not response_format)
    call_args = mock_client.chat.completions.create.call_args
    api_params = call_args[1] if call_args[1] else call_args[0]
    
    # HuggingFace doesn't use response_format, so it shouldn't be present
    self.assertNotIn('response_format', api_params)
    
    # Verify prompt was modified to request JSON
    expected_content = 'Test prompt\n\nPlease respond in valid JSON format.'
    self.assertEqual(api_params['messages'][0]['content'], expected_content)

  @mock.patch('openai.OpenAI')
  def test_hf_api_error_handling(self, mock_openai):
    """Test API error handling."""
    mock_client = mock.Mock()
    mock_openai.return_value = mock_client
    
    # Mock an API error
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    
    model = inference.HFLanguageModel(api_key="test_token")
    
    with self.assertRaises(inference.InferenceOutputError) as cm:
      model._process_single_prompt("Test prompt", {})
    
    self.assertIn('HuggingFace API error', str(cm.exception))
    self.assertIn('API Error', str(cm.exception))

  @mock.patch('openai.OpenAI')
  def test_hf_parse_output_json(self, mock_openai):
    """Test JSON output parsing."""
    mock_openai.return_value = mock.Mock()
    
    from langextract import data
    model = inference.HFLanguageModel(
        api_key="test_token",
        format_type=data.FormatType.JSON
    )
    
    json_output = '{"name": "test", "value": 42}'
    parsed = model.parse_output(json_output)
    
    expected = {"name": "test", "value": 42}
    self.assertEqual(parsed, expected)

  @mock.patch('openai.OpenAI')
  def test_hf_parse_output_yaml(self, mock_openai):
    """Test YAML output parsing."""
    mock_openai.return_value = mock.Mock()
    
    from langextract import data
    model = inference.HFLanguageModel(
        api_key="test_token",
        format_type=data.FormatType.YAML
    )
    
    yaml_output = 'name: test\nvalue: 42'
    parsed = model.parse_output(yaml_output)
    
    expected = {"name": "test", "value": 42}
    self.assertEqual(parsed, expected)


if __name__ == "__main__":
  absltest.main()
