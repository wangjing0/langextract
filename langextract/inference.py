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

import anthropic
import openai
import requests
from typing_extensions import override
import yaml



from langextract import data
from langextract import schema


_OLLAMA_DEFAULT_MODEL_URL = 'http://localhost:11434'


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
  claude_schema: schema.ClaudeSchema | None = None
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
      claude_schema: schema.ClaudeSchema | None = None,
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
      claude_schema: Optional schema for structured output.
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
    self.claude_schema = claude_schema
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
      if self.claude_schema:
        schema_instruction = f"\n\nPlease respond in valid JSON format matching this schema: {self.claude_schema.schema_dict}"
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
class GPTLanguageModel(BaseLanguageModel):
  """Language model inference using OpenAI's GPT API with structured output."""

  model_id: str = 'gpt-4o-mini'
  api_key: str | None = None
  openai_schema: dict[str, Any] | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float = 0.0
  seed: int | None = None
  max_workers: int = 10
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model_id: str = 'gpt-4o-mini',
      api_key: str | None = None,
      openai_schema: dict[str, Any] | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      seed: int | None = None,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the GPT language model.

    Args:
      model_id: The GPT model ID to use.
      api_key: API key for OpenAI service.
      openai_schema: Optional JSON schema for structured output.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      seed: Random seed for deterministic generation.
      max_workers: Maximum number of parallel API calls.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising ``TypeError``.
    """
    self.model_id = model_id
    self.api_key = api_key
    self.openai_schema = openai_schema
    self.format_type = format_type
    self.temperature = temperature
    self.seed = seed
    self.max_workers = max_workers
    self._extra_kwargs = kwargs or {}

    if not self.api_key:
      raise ValueError('API key not provided.')

    self._client = openai.OpenAI(api_key=self.api_key)

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

  def _process_single_prompt(self, prompt: str, config: dict) -> ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    try:
      # Build API call parameters
      api_params = {
          'model': self.model_id,
          'max_tokens': config.get('max_output_tokens', 1024),
          'temperature': config.get('temperature', self.temperature),
          'messages': [{'role': 'user', 'content': prompt}]
      }
      
      # Add seed if provided
      seed_value = config.get('seed', self.seed)
      if seed_value is not None:
        api_params['seed'] = seed_value

      # Add other optional parameters
      if 'top_p' in config:
        api_params['top_p'] = config['top_p']

      # Handle structured output with JSON schema
      if self.openai_schema:
        api_params['response_format'] = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'structured_output',
                'schema': self.openai_schema
            }
        }
      elif self.format_type == data.FormatType.JSON:
        api_params['response_format'] = {'type': 'json_object'}
        # Add JSON format instruction to prompt
        if 'respond in valid JSON format' not in prompt.lower():
          prompt = prompt + '\n\nPlease respond in valid JSON format.'
          api_params['messages'][0]['content'] = prompt
      
      response = self._client.chat.completions.create(**api_params)

      return ScoredOutput(score=1.0, output=response.choices[0].message.content)

    except Exception as e:
      raise InferenceOutputError(f'OpenAI API error: {str(e)}') from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    """Runs inference on a list of prompts via OpenAI's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, seed, etc.)

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
    """Parses GPT output as JSON or YAML."""
    try:
      if self.format_type == data.FormatType.JSON:
        return json.loads(output)
      else:
        return yaml.safe_load(output)
    except Exception as e:
      raise ValueError(
          f'Failed to parse output as {self.format_type.name}: {str(e)}'
      ) from e
