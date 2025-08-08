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

"""Simple library for performing language model inference."""

import abc
from collections.abc import Iterator, Mapping, Sequence
import concurrent.futures
import dataclasses
import enum
import json
import textwrap
from typing import Any
import google.genai as genai
import anthropic
import openai
import requests
from typing_extensions import override
import yaml
import re
import logging

from langextract import data
from langextract import schema

logging.basicConfig(level=logging.DEBUG)

_OLLAMA_DEFAULT_MODEL_URL = 'http://localhost:11434'
_HF_DEFAULT_MODEL_URL = 'https://router.huggingface.co/v1'


@dataclasses.dataclass(frozen=True)
class ScoredOutput:
  """Scored output."""

  score: float | None = None
  output: str | None = None

  def __str__(self) -> str:
    if self.output is None:
      return f'Score: {self.score:.2f}\nOutput: None'
    formatted_lines = textwrap.indent(self.output, prefix='  ')
    return f'Score: {self.score:.2f}\nOutput:\n{formatted_lines}'


class InferenceOutputError(Exception):
  """Exception raised when no scored outputs are available from the language model."""

  def __init__(self, message: str):
    self.message = message
    super().__init__(self.message)


class BaseLanguageModel(abc.ABC):
  """An abstract inference class for managing LLM inference.

  Attributes:
    _constraint: A `Constraint` object specifying constraints for model output.
  """

  def __init__(self, constraint: schema.Constraint = schema.Constraint()):
    """Initializes the BaseLanguageModel with an optional constraint.

    Args:
      constraint: Applies constraints when decoding the output. Defaults to no
        constraint.
    """
    self._constraint = constraint

  @abc.abstractmethod
  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    """Implements language model inference.

    Args:
      batch_prompts: Batch of inputs for inference. Single element list can be
        used for a single input.
      **kwargs: Additional arguments for inference, like temperature and
        max_decode_steps.

    Returns: Batch of Sequence of probable output text outputs, sorted by
      descending
      score.
    """


class InferenceType(enum.Enum):
  ITERATIVE = 'iterative'
  MULTIPROCESS = 'multiprocess'



@dataclasses.dataclass(init=False)
class OllamaLanguageModel(BaseLanguageModel):
  """Language model inference class using Ollama based host."""

  _model: str
  _model_url: str
  _structured_output_format: str
  _constraint: schema.Constraint = dataclasses.field(
      default_factory=schema.Constraint, repr=False, compare=False
  )
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model: str,
      model_url: str = _OLLAMA_DEFAULT_MODEL_URL,
      structured_output_format: str = 'json',
      constraint: schema.Constraint = schema.Constraint(),
      **kwargs,
  ) -> None:
    self._model = model
    self._model_url = model_url
    self._structured_output_format = structured_output_format
    self._constraint = constraint
    self._extra_kwargs = kwargs or {}
    super().__init__(constraint=constraint)

  @override
  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    for prompt in batch_prompts:
      response = self._ollama_query(
          prompt=prompt,
          model=self._model,
          structured_output_format=self._structured_output_format,
          model_url=self._model_url,
      )
      # No score for Ollama. Default to 1.0
      yield [ScoredOutput(score=1.0, output=response['response'])]

  def _ollama_query(
      self,
      prompt: str,
      model: str = 'gemma2:latest',
      temperature: float = 0.8,
      seed: int | None = None,
      top_k: int | None = None,
      max_output_tokens: int | None = None,
      structured_output_format: str | None = None,  # like 'json'
      system: str = '',
      raw: bool = False,
      model_url: str = _OLLAMA_DEFAULT_MODEL_URL,
      timeout: int = 30,  # seconds
      keep_alive: int = 5 * 60,  # if loading, keep model up for 5 minutes.
      num_threads: int | None = None,
      num_ctx: int = 2048,
  ) -> Mapping[str, Any]:
    """Sends a prompt to an Ollama model and returns the generated response.

    This function makes an HTTP POST request to the `/api/generate` endpoint of
    an Ollama server. It can optionally load the specified model first, generate
    a response (with or without streaming), then return a parsed JSON response.

    Args:
      prompt: The text prompt to send to the model.
      model: The name of the model to use, e.g. "gemma2:latest".
      temperature: Sampling temperature. Higher values produce more diverse
        output.
      seed: Seed for reproducible generation. If None, random seed is used.
      top_k: The top-K parameter for sampling.
      max_output_tokens: Maximum tokens to generate. If None, the model's
        default is used.
      structured_output_format: If set to "json" or a JSON schema dict, requests
        structured outputs from the model. See Ollama documentation for details.
      system: A system prompt to override any system-level instructions.
      raw: If True, bypasses any internal prompt templating; you provide the
        entire raw prompt.
      model_url: The base URL for the Ollama server, typically
        "http://localhost:11434".
      timeout: Timeout (in seconds) for the HTTP request.
      keep_alive: How long (in seconds) the model remains loaded after
        generation completes.
      num_threads: Number of CPU threads to use. If None, Ollama uses a default
        heuristic.
      num_ctx: Number of context tokens allowed. If None, uses model’s default
        or config.

    Returns:
      A mapping (dictionary-like) containing the server’s JSON response. For
      non-streaming calls, the `"response"` key typically contains the entire
      generated text.

    Raises:
      ValueError: If the server returns a 404 (model not found) or any non-OK
      status code other than 200. Also raised on read timeouts or other request
      exceptions.
    """
    options = {'keep_alive': keep_alive}
    if seed:
      options['seed'] = seed
    if temperature:
      options['temperature'] = temperature
    if top_k:
      options['top_k'] = top_k
    if num_threads:
      options['num_thread'] = num_threads
    if max_output_tokens:
      options['num_predict'] = max_output_tokens
    if num_ctx:
      options['num_ctx'] = num_ctx
    model_url = model_url + '/api/generate'

    payload = {
        'model': model,
        'prompt': prompt,
        'system': system,
        'stream': False,
        'raw': raw,
        'format': structured_output_format,
        'options': options,
    }
    try:
      response = requests.post(
          model_url,
          headers={
              'Content-Type': 'application/json',
              'Accept': 'application/json',
          },
          json=payload,
          timeout=timeout,
      )
    except requests.exceptions.RequestException as e:
      if isinstance(e, requests.exceptions.ReadTimeout):
        msg = (
            f'Ollama Model timed out (timeout={timeout},'
            f' num_threads={num_threads})'
        )
        raise ValueError(msg) from e
      raise e

    response.encoding = 'utf-8'
    if response.status_code == 200:
      return response.json()
    if response.status_code == 404:
      raise ValueError(
          f"Can't find Ollama {model}. Try launching `ollama run {model}`"
          ' from command line.'
      )
    else:
      raise ValueError(
          f'Ollama model failed with status code {response.status_code}.'
      )


@dataclasses.dataclass(init=False)
class ClaudeLanguageModel(BaseLanguageModel):
  """Language model inference using Anthropic's Claude API with structured output."""

  model_id: str = 'claude-3-5-haiku-latest'
  api_key: str | None = None
  structured_schema: schema.StructuredSchema | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float = 0.0
  seed: int | None = None
  max_workers: int = 10
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model_id: str = 'claude-3-5-haiku-latest',
      api_key: str | None = None,
      structured_schema: schema.StructuredSchema | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      seed: int | None = None,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the Claude language model.

    Args:
      model_id: The Claude model ID to use.
      api_key: API key for Claude service.
      structured_schema: Optional StructuredSchema for structured output.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      seed: Random seed for deterministic generation. Currently has no effect
        as the Anthropic API does not support seed parameters (as of Jan 2025).
        Stored for future compatibility.
      max_workers: Maximum number of parallel API calls.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising ``TypeError``.
    """
    self.model_id = model_id
    self.api_key = api_key
    self.structured_schema = structured_schema
    self.format_type = format_type
    self.temperature = temperature
    self.seed = seed
    self.max_workers = max_workers
    self._extra_kwargs = kwargs or {}

    if not self.api_key:
      raise ValueError('API key not provided.')

    self._client = anthropic.Anthropic(api_key=self.api_key)

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

  def _process_single_prompt(self, prompt: str, config: dict) -> ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    try:
      # For structured output with Claude, we'll add instructions to the prompt
      if self.structured_schema:
        schema_instruction = f"\n\nPlease respond in valid JSON format matching this schema: {self.structured_schema.claude_schema}"
        prompt = prompt + schema_instruction

      # Build API call parameters
      api_params = {
          'model': self.model_id,
          'max_tokens': config.get('max_output_tokens', 1024),
          'temperature': config.get('temperature', self.temperature),
          'messages': [{'role': 'user', 'content': prompt}]
      }
      
      # Add seed if provided
      # Note: As of January 2025, Anthropic Claude API does not support seed parameters
      # This parameter is stored for future compatibility but currently has no effect
      seed_value = config.get('seed', self.seed)
      if seed_value is not None:
        # Log that seed is requested but not supported
        import logging
        logging.debug(f"Seed parameter {seed_value} requested but not supported by current Anthropic API")
      
      response = self._client.messages.create(**api_params)

      return ScoredOutput(score=1.0, output=response.content[0].text)

    except Exception as e:
      raise InferenceOutputError(f'Claude API error: {str(e)}') from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    """Runs inference on a list of prompts via Claude's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, top_k, etc.)

    Yields:
      Lists of ScoredOutputs.
    """
    config = {
        'temperature': kwargs.get('temperature', self.temperature),
    }
    if 'max_output_tokens' in kwargs:
      config['max_output_tokens'] = kwargs['max_output_tokens']
    if 'seed' in kwargs:
      config['seed'] = kwargs['seed']
    if 'top_p' in kwargs:
      config['top_p'] = kwargs['top_p']
    if 'top_k' in kwargs:
      config['top_k'] = kwargs['top_k']

    # Use parallel processing for batches larger than 1
    if len(batch_prompts) > 1 and self.max_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(self.max_workers, len(batch_prompts))
      ) as executor:
        future_to_index = {
            executor.submit(
                self._process_single_prompt, prompt, config.copy()
            ): i
            for i, prompt in enumerate(batch_prompts)
        }

        results: list[ScoredOutput | None] = [None] * len(batch_prompts)
        for future in concurrent.futures.as_completed(future_to_index):
          index = future_to_index[future]
          try:
            results[index] = future.result()
          except Exception as e:
            raise InferenceOutputError(
                f'Parallel inference error: {str(e)}'
            ) from e

        for result in results:
          if result is None:
            raise InferenceOutputError('Failed to process one or more prompts')
          yield [result]
    else:
      # Sequential processing for single prompt or worker
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]

  def parse_output(self, output: str) -> Any:
    """Parses Claude output as JSON or YAML."""
    try:
      if self.format_type == data.FormatType.JSON:
        return json.loads(output)
      else:
        return yaml.safe_load(output)
    except Exception as e:
      raise ValueError(
          f'Failed to parse output as {self.format_type.name}: {str(e)}'
      ) from e



@dataclasses.dataclass(init=False)
class GeminiLanguageModel(BaseLanguageModel):
  """Language model inference using Google's Gemini API with structured output."""

  model_id: str = 'gemini-2.5-flash'
  api_key: str | None = None
  structured_schema: schema.StructuredSchema | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float = 0.0
  max_workers: int = 10
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model_id: str = 'gemini-2.5-flash',
      api_key: str | None = None,
      structured_schema: schema.StructuredSchema | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the Gemini language model.

    Args:
      model_id: The Gemini model ID to use.
      api_key: API key for Gemini service.
      structured_schema: Optional StructuredSchema for structured output.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising ``TypeError``.
    """
    self.model_id = model_id
    self.api_key = api_key
    self.structured_schema = structured_schema
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self._extra_kwargs = kwargs or {}

    if not self.api_key:
      raise ValueError('API key not provided.')

    self._client = genai.Client(api_key=self.api_key)

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

  def _process_single_prompt(self, prompt: str, config: dict) -> ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    try:
      if self.structured_schema:
        response_schema = self.structured_schema.schema_dict
        mime_type = (
            'application/json'
            if self.format_type == data.FormatType.JSON
            else 'application/yaml'
        )
        config['response_mime_type'] = mime_type
        config['response_schema'] = response_schema

      response = self._client.models.generate_content(
          model=self.model_id, contents=prompt, config=config
      )

      return ScoredOutput(score=1.0, output=response.text)

    except Exception as e:
      raise InferenceOutputError(f'Gemini API error: {str(e)}') from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    """Runs inference on a list of prompts via Gemini's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, top_k, etc.)

    Yields:
      Lists of ScoredOutputs.
    """
    config = {
        'temperature': kwargs.get('temperature', self.temperature),
    }
    if 'max_output_tokens' in kwargs:
      config['max_output_tokens'] = kwargs['max_output_tokens']
    if 'top_p' in kwargs:
      config['top_p'] = kwargs['top_p']
    if 'top_k' in kwargs:
      config['top_k'] = kwargs['top_k']

    # Use parallel processing for batches larger than 1
    if len(batch_prompts) > 1 and self.max_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(self.max_workers, len(batch_prompts))
      ) as executor:
        future_to_index = {
            executor.submit(
                self._process_single_prompt, prompt, config.copy()
            ): i
            for i, prompt in enumerate(batch_prompts)
        }

        results: list[ScoredOutput | None] = [None] * len(batch_prompts)
        for future in concurrent.futures.as_completed(future_to_index):
          index = future_to_index[future]
          try:
            results[index] = future.result()
          except Exception as e:
            raise InferenceOutputError(
                f'Parallel inference error: {str(e)}'
            ) from e

        for result in results:
          if result is None:
            raise InferenceOutputError('Failed to process one or more prompts')
          yield [result]
    else:
      # Sequential processing for single prompt or worker
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]

  def parse_output(self, output: str) -> Any:
    """Parses Gemini output as JSON or YAML.

    Note: This expects raw JSON/YAML without code fences.
    Code fence extraction is handled by resolver.py.
    """
    try:
      if self.format_type == data.FormatType.JSON:
        return json.loads(output)
      else:
        return yaml.safe_load(output)
    except Exception as e:
      raise ValueError(
          f'Failed to parse output as {self.format_type.name}: {str(e)}'
      ) from e


@dataclasses.dataclass(init=False)
class OpenAILanguageModel(BaseLanguageModel):
  """Language model inference using OpenAI's API with structured output."""

  model_id: str = 'gpt-4o-mini'
  api_key: str | None = None
  organization: str | None = None
  structured_schema: schema.StructuredSchema | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float = 0.0
  max_workers: int = 10
  _client: openai.OpenAI | None = dataclasses.field(
      default=None, repr=False, compare=False
  )
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model_id: str = 'gpt-4o-mini',
      api_key: str | None = None,
      organization: str | None = None,
      structured_schema: schema.StructuredSchema | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the OpenAI language model.

    Args:
      model_id: The OpenAI model ID to use (e.g., 'gpt-4o-mini', 'gpt-4o').
      api_key: API key for OpenAI service.
      organization: Optional OpenAI organization ID.
      structured_schema: Optional StructuredSchema for structured output.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising ``TypeError``.
    """
    self.model_id = model_id
    self.api_key = api_key
    self.organization = organization
    self.structured_schema = structured_schema
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self._extra_kwargs = kwargs or {}

    if not self.api_key:
      raise ValueError('API key not provided.')

    # Initialize the OpenAI client
    self._client = openai.OpenAI(
        api_key=self.api_key, 
        organization=self.organization
    )

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

  def _process_single_prompt(self, prompt: str, config: dict) -> ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    try:
      # Build API call parameters
      if self.model_id.startswith('gpt-5'):
          # GPT-5 models use different message format with developer role
          api_params = {
              'model': self.model_id,
              'messages': [
                  {
                      'role': 'developer',
                      'content': [
                          {
                              'type': 'text',
                              'text': 'You are an expert entity extraction assistant. Extract entities and relationships as requested.'
                          }
                      ]
                  },
                  {
                      'role': 'user',
                      'content': [{'type': 'text', 'text': prompt}]
                  }
              ],
              'reasoning_effort': 'medium'
          }
      else:
          api_params = {
              'model': self.model_id,
              'messages': [{'role': 'user', 'content': prompt}]
          }
      
      # Use max_completion_tokens for GPT-5 models, max_tokens for others
      max_tokens_value = config.get('max_output_tokens', 1024)
      if self.model_id.startswith('gpt-5'):
          # GPT-5-nano has limited output capacity, be conservative
          api_params['max_completion_tokens'] = 8000
          # GPT-5 does not support temperature
      else:
          api_params['max_tokens'] = max_tokens_value
          api_params['temperature'] = config.get('temperature', self.temperature)
      
      # Add optional parameters
      if 'top_p' in config:
        api_params['top_p'] = config['top_p']

      # Handle structured output with JSON schema
      if self.structured_schema:
        api_params['response_format'] = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'structured_output',
                'schema': self.structured_schema.openai_schema
            }
        }
      elif self.format_type == data.FormatType.JSON:
        # Use json_object for GPT-5, regular json_object for others
        api_params['response_format'] = {'type': 'json_object'}
        # Add JSON format instruction to prompt
        if 'respond in valid JSON format' not in prompt.lower():
          prompt = prompt + '\n\nPlease respond in valid JSON format.'
          # Update prompt in the correct message format
          if self.model_id.startswith('gpt-5'):
              api_params['messages'][1]['content'][0]['text'] = prompt  # Update user message
          else:
              api_params['messages'][0]['content'] = prompt
      
      # Create the chat completion using the v1.x client API
      response = self._client.chat.completions.create(**api_params)

      # Extract the response text using the v1.x response format
      output_text = response.choices[0].message.content
      
      # Handle GPT-5 output limitations
      if self.model_id.startswith('gpt-5'):
          # GPT-5-nano has severe output limitations, handle gracefully
          if (not output_text or output_text.strip() == '') and response.choices[0].finish_reason == 'length':
              # Return a structured error message that can be parsed
              output_text = '{"error": "GPT-5-nano output capacity exceeded", "message": "Consider using shorter prompts or fewer examples"}'

      return ScoredOutput(score=1.0, output=output_text)

    except Exception as e:
      raise InferenceOutputError(f'OpenAI API error: {str(e)}') from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    """Runs inference on a list of prompts via OpenAI's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, etc.)

    Yields:
      Lists of ScoredOutputs.
    """
    config = {
        'temperature': kwargs.get('temperature', self.temperature),
    }
    if 'max_output_tokens' in kwargs:
      config['max_output_tokens'] = kwargs['max_output_tokens']
    if 'top_p' in kwargs:
      config['top_p'] = kwargs['top_p']

    # Use parallel processing for batches larger than 1
    if len(batch_prompts) > 1 and self.max_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(self.max_workers, len(batch_prompts))
      ) as executor:
        future_to_index = {
            executor.submit(
                self._process_single_prompt, prompt, config.copy()
            ): i
            for i, prompt in enumerate(batch_prompts)
        }

        results: list[ScoredOutput | None] = [None] * len(batch_prompts)
        for future in concurrent.futures.as_completed(future_to_index):
          index = future_to_index[future]
          try:
            results[index] = future.result()
          except Exception as e:
            raise InferenceOutputError(
                f'Parallel inference error: {str(e)}'
            ) from e

        for result in results:
          if result is None:
            raise InferenceOutputError('Failed to process one or more prompts')
          yield [result]
    else:
      # Sequential processing for single prompt or worker
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]

  def parse_output(self, output: str) -> Any:
    """Parses OpenAI output as JSON or YAML.

    Note: This expects raw JSON/YAML without code fences.
    Code fence extraction is handled by resolver.py.
    """
    try:
      if self.format_type == data.FormatType.JSON:
        return json.loads(output)
      else:
        return yaml.safe_load(output)
    except Exception as e:
      raise ValueError(
          f'Failed to parse output as {self.format_type.name}: {str(e)}'
      ) from e
  

@dataclasses.dataclass(init=False)
class HFLanguageModel(BaseLanguageModel):
  """Language model inference using HuggingFace's OpenAI-compatible API."""

  model_id: str = 'openai/gpt-oss-120b:cerebras'
  api_key: str | None = None
  base_url: str = _HF_DEFAULT_MODEL_URL
  structured_schema: schema.StructuredSchema | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float = 0.0
  max_workers: int = 10
  _client: openai.OpenAI | None = dataclasses.field(
      default=None, repr=False, compare=False
  )
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model_id: str = 'openai/gpt-oss-120b:cerebras',
      api_key: str | None = None,
      base_url: str = 'https://router.huggingface.co/v1',
      structured_schema: schema.StructuredSchema | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the HuggingFace language model.

    Args:
      model_id: The HuggingFace model ID to use (e.g., 'openai/gpt-oss-120b:cerebras').
      api_key: HuggingFace token (HF_TOKEN environment variable).
      base_url: Base URL for HuggingFace's OpenAI-compatible API.
      structured_schema: Optional StructuredSchema for structured output.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising ``TypeError``.
    """
    self.model_id = model_id
    self.api_key = api_key
    self.base_url = base_url
    self.structured_schema = structured_schema
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self._extra_kwargs = kwargs or {}

    if not self.api_key:
      import os
      self.api_key = os.getenv('HF_TOKEN')
      if not self.api_key:
        raise ValueError('HF_TOKEN not provided in api_key parameter or environment variable.')

    # Initialize the OpenAI client with HuggingFace base URL
    self._client = openai.OpenAI(
        base_url=self.base_url,
        api_key=self.api_key
    )

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

  def _process_single_prompt(self, prompt: str, config: dict) -> ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    try:
      # Build API call parameters
      api_params = {
          'model': self.model_id,
          'max_tokens': config.get('max_output_tokens', 4096),  # Increase default for HF models
          'temperature': config.get('temperature', self.temperature),
          'messages': [{'role': 'user', 'content': prompt}]
      }
      
      # Add optional parameters
      if 'top_p' in config:
        api_params['top_p'] = config['top_p']

      # Handle structured output - HuggingFace doesn't support response_format
      # Instead, we add instructions to the prompt
      if self.structured_schema:
        schema_instruction = f"\n\nPlease respond in valid JSON format matching this schema: {self.structured_schema.openai_schema}. Wrap your response in ```json and ```"
        prompt = prompt + schema_instruction
        api_params['messages'][0]['content'] = prompt
      elif self.format_type == data.FormatType.JSON:
        # Add JSON format instruction to prompt
        if 'respond in valid JSON format' not in prompt.lower():
          prompt = prompt + '\n\nPlease respond in valid JSON format, wrapped in ```json and ```'
          api_params['messages'][0]['content'] = prompt
      
      # Create the chat completion using the OpenAI client
      response = self._client.chat.completions.create(**api_params)

      # Extract the response text - be more careful with regex cleanup
      raw_output = response.choices[0].message.content
      if raw_output:
        # Remove ```json at the beginning and ``` at the end more carefully
        output_text = re.sub(r'^```json\s*', '', raw_output.strip())
        output_text = re.sub(r'\s*```$', '', output_text.strip())
      else:
        output_text = raw_output
      

      return ScoredOutput(score=1.0, output=output_text)

    except Exception as e:
      raise InferenceOutputError(f'HuggingFace API error: {str(e)}') from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    """Runs inference on a list of prompts via HuggingFace's OpenAI-compatible API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, etc.)

    Yields:
      Lists of ScoredOutputs.
    """
    config = {
        'temperature': kwargs.get('temperature', self.temperature),
    }
    if 'max_output_tokens' in kwargs:
      config['max_output_tokens'] = kwargs['max_output_tokens']
    if 'top_p' in kwargs:
      config['top_p'] = kwargs['top_p']

    # Use parallel processing for batches larger than 1
    if len(batch_prompts) > 1 and self.max_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(self.max_workers, len(batch_prompts))
      ) as executor:
        future_to_index = {
            executor.submit(
                self._process_single_prompt, prompt, config.copy()
            ): i
            for i, prompt in enumerate(batch_prompts)
        }

        results: list[ScoredOutput | None] = [None] * len(batch_prompts)
        for future in concurrent.futures.as_completed(future_to_index):
          index = future_to_index[future]
          try:
            results[index] = future.result()
          except Exception as e:
            raise InferenceOutputError(
                f'Parallel inference error: {str(e)}'
            ) from e

        for result in results:
          if result is None:
            raise InferenceOutputError('Failed to process one or more prompts')
          yield [result]
    else:
      # Sequential processing for single prompt or worker
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]

  def parse_output(self, output: str) -> Any:
    """Parses HuggingFace output as JSON or YAML.

    Note: This expects raw JSON/YAML without code fences.
    Code fence extraction is handled by resolver.py.
    """
    try:
      if self.format_type == data.FormatType.JSON:
        return json.loads(output)
      else:
        return yaml.safe_load(output)
    except Exception as e:
      raise ValueError(
          f'Failed to parse output as {self.format_type.name}: {str(e)}'
      ) from e