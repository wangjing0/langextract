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
import langfun as lf
from absl.testing import absltest
from langextract import inference


class TestLangFunLanguageModel(absltest.TestCase):
  @mock.patch.object(
      inference.lf.core.language_model, "LanguageModel", autospec=True
  )
  def test_langfun_infer(self, mock_lf_model):
    mock_client_instance = mock_lf_model.return_value
    metadata = {
        "score": -0.004259720362824737,
        "logprobs": None,
        "is_cached": False,
    }
    source = lf.UserMessage(
        text="What's heart in Italian?.",
        sender="User",
        metadata={"formatted_text": "What's heart in Italian?."},
        tags=["lm-input"],
    )
    sample = lf.LMSample(
        response=lf.AIMessage(
            text="Cuore",
            sender="AI",
            metadata=metadata,
            source=source,
            tags=["lm-response"],
        ),
        score=-0.004259720362824737,
    )
    actual_response = lf.LMSamplingResult(
        samples=[sample],
    )

    # Mock the sample response.
    mock_client_instance.sample.return_value = [actual_response]
    model = inference.LangFunLanguageModel(language_model=mock_client_instance)

    batch_prompts = ["What's heart in Italian?"]

    expected_results = [
        [inference.ScoredOutput(score=-0.004259720362824737, output="Cuore")]
    ]

    results = list(model.infer(batch_prompts))

    mock_client_instance.sample.assert_called_once_with(prompts=batch_prompts)
    self.assertEqual(results, expected_results)


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


if __name__ == "__main__":
  absltest.main()
