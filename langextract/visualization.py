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

"""Utility functions for visualizing LangExtract extractions in notebooks.

Example
-------
>>> import langextract as lx
>>> doc = lx.extract(...)
>>> lx.visualize(doc)
"""

from __future__ import annotations

import dataclasses
import enum
import html
import itertools
import json
import textwrap

import os
import pathlib
from langextract import data as _data
from langextract import io as _io
from langextract import progress

# Fallback if IPython is not present
try:
  from IPython.display import HTML  # type: ignore
except Exception:
  HTML = None  # pytype: disable=annotation-type-mismatch

_PALETTE: list[str] = [
    '#D2E3FC',  # Light Blue (Primary Container)
    '#C8E6C9',  # Light Green (Tertiary Container)
    '#FEF0C3',  # Light Yellow (Primary Color)
    '#F9DEDC',  # Light Red (Error Container)
    '#FFDDBE',  # Light Orange (Tertiary Container)
    '#EADDFF',  # Light Purple (Secondary/Tertiary Container)
    '#C4E9E4',  # Light Teal (Teal Container)
    '#FCE4EC',  # Light Pink (Pink Container)
    '#E8EAED',  # Very Light Grey (Neutral Highlight)
    '#DDE8E8',  # Pale Cyan (Cyan Container)
]

_VISUALIZATION_CSS = textwrap.dedent("""\
    <style>
    .lx-highlight { position: relative; border-radius:3px; padding:1px 2px;}
    .lx-highlight .lx-tooltip {
      visibility: hidden;
      opacity: 0;
      transition: opacity 0.2s ease-in-out;
      background: #333;
      color: #fff;
      text-align: left;
      border-radius: 4px;
      padding: 6px 8px;
      position: absolute;
      z-index: 1000;
      bottom: 125%;
      left: 50%;
      transform: translateX(-50%);
      font-size: 12px;
      max-width: 240px;
      white-space: normal;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    .lx-highlight:hover .lx-tooltip { visibility: visible; opacity:1; }
    .lx-animated-wrapper { max-width: 100%; font-family: Arial, sans-serif; }
    .lx-controls {
      background: #fafafa; border: 1px solid #90caf9; border-radius: 8px;
      padding: 12px; margin-bottom: 16px;
    }
    .lx-button-row {
      display: flex; justify-content: center; gap: 8px; margin-bottom: 12px;
    }
    .lx-control-btn {
      background: #4285f4; color: white; border: none; border-radius: 4px;
      padding: 8px 16px; cursor: pointer; font-size: 13px; font-weight: 500;
      transition: background-color 0.2s;
    }
    .lx-control-btn:hover { background: #3367d6; }
    .lx-progress-container {
      margin-bottom: 8px;
    }
    .lx-progress-slider {
      width: 100%; margin: 0; appearance: none; height: 6px;
      background: #ddd; border-radius: 3px; outline: none;
    }
    .lx-progress-slider::-webkit-slider-thumb {
      appearance: none; width: 18px; height: 18px; background: #4285f4;
      border-radius: 50%; cursor: pointer;
    }
    .lx-progress-slider::-moz-range-thumb {
      width: 18px; height: 18px; background: #4285f4; border-radius: 50%;
      cursor: pointer; border: none;
    }
    .lx-status-text {
      text-align: center; font-size: 12px; color: #666; margin-top: 4px;
    }
    .lx-text-window {
      font-family: monospace; white-space: pre-wrap; border: 1px solid #90caf9;
      padding: 12px; height: 60vh; min-height: 200px; max-height: 80vh; 
      overflow-y: auto; margin-bottom: 12px; line-height: 1.6; resize: vertical;
    }
    .lx-attributes-panel {
      background: #fafafa; border: 1px solid #90caf9; border-radius: 6px;
      padding: 8px 10px; margin-top: 8px; font-size: 13px;
    }
    .lx-current-highlight {
      text-decoration: underline;
      text-decoration-color: #ff4444;
      text-decoration-thickness: 3px;
      font-weight: bold;
      animation: lx-pulse 1s ease-in-out;
    }
    @keyframes lx-pulse {
      0% { text-decoration-color: #ff4444; }
      50% { text-decoration-color: #ff0000; }
      100% { text-decoration-color: #ff4444; }
    }
    .lx-legend { 
      font-size: 12px; margin-bottom: 8px; 
      padding-bottom: 8px; border-bottom: 1px solid #e0e0e0; 
    }
    .lx-label {
      display: inline-block;
      padding: 2px 4px;
      border-radius: 3px;
      margin-right: 4px;
      color: #000;
    }
    .lx-attr-key {
      font-weight: 600;
      color: #1565c0;
      letter-spacing: 0.3px;
    }
    .lx-attr-value {
      font-weight: 400;
      opacity: 0.85;
      letter-spacing: 0.2px;
    }

    /* Add optimizations with larger fonts and better readability for GIFs */
    .lx-gif-optimized .lx-text-window { font-size: 16px; line-height: 1.8; }
    .lx-gif-optimized .lx-attributes-panel { font-size: 15px; }
    .lx-gif-optimized .lx-current-highlight { text-decoration-thickness: 4px; }
    </style>""")


def _assign_colors(extractions: list[_data.Extraction]) -> dict[str, str]:
  """Assigns a background colour to each extraction class.

  Args:
    extractions: list of extractions.

  Returns:
    Mapping from extraction_class to a hex colour string.
  """
  classes = {e.extraction_class for e in extractions if e.char_interval}
  color_map: dict[str, str] = {}
  palette_cycle = itertools.cycle(_PALETTE)
  for cls in sorted(classes):
    color_map[cls] = next(palette_cycle)
  return color_map


def _filter_valid_extractions(
    extractions: list[_data.Extraction],
) -> list[_data.Extraction]:
  """Filters extractions to only include those with valid char intervals."""
  return [
      e
      for e in extractions
      if (
          e.char_interval
          and e.char_interval.start_pos is not None
          and e.char_interval.end_pos is not None
      )
  ]


class TagType(enum.Enum):
  """Enum for span boundary tag types."""

  START = 'start'
  END = 'end'


@dataclasses.dataclass(frozen=True)
class SpanPoint:
  """Represents a span boundary point for HTML generation.

  Attributes:
    position: Character position in the text.
    tag_type: Type of span boundary (START or END).
    span_idx: Index of the span for HTML data-idx attribute.
    extraction: The extraction data associated with this span.
  """

  position: int
  tag_type: TagType
  span_idx: int
  extraction: _data.Extraction


def _build_highlighted_text(
    text: str,
    extractions: list[_data.Extraction],
    color_map: dict[str, str],
) -> str:
  """Returns text with <span> highlights inserted, supporting nesting.

  Args:
    text: Original document text.
    extractions: List of extraction objects with char_intervals.
    color_map: Mapping of extraction_class to colour.
  """
  # Sort extractions by position to ensure proper indexing
  def _sort_key(extraction):
    start = extraction.char_interval.start_pos
    end = extraction.char_interval.end_pos
    span_length = end - start
    return (start, -span_length)  # longer spans first
  
  sorted_extractions = sorted(extractions, key=_sort_key)
  
  points = []
  span_lengths = {}
  for index, extraction in enumerate(sorted_extractions):
    if (
        not extraction.char_interval
        or extraction.char_interval.start_pos is None
        or extraction.char_interval.end_pos is None
        or extraction.char_interval.start_pos
        >= extraction.char_interval.end_pos
    ):
      continue

    start_pos = extraction.char_interval.start_pos
    end_pos = extraction.char_interval.end_pos
    # Use index directly since extractions are already sorted in the desired order
    points.append(SpanPoint(start_pos, TagType.START, index, extraction))
    points.append(SpanPoint(end_pos, TagType.END, index, extraction))
    span_lengths[index] = end_pos - start_pos

  def sort_key(point: SpanPoint):
    """Sorts span boundary points for proper HTML nesting.

    Sorts by position first, then handles ties at the same position to ensure
    proper HTML nesting. At the same position:
    1. End tags come before start tags (to close before opening)
    2. Among end tags: shorter spans close first
    3. Among start tags: longer spans open first

    Args:
      point: SpanPoint containing position, tag_type, span_idx, and extraction.

    Returns:
      Sort key tuple ensuring proper nesting order.
    """
    span_length = span_lengths.get(point.span_idx, 0)

    if point.tag_type == TagType.END:
      return (point.position, 0, span_length)
    else:  # point.tag_type == TagType.START
      return (point.position, 1, -span_length)

  points.sort(key=sort_key)

  html_parts: list[str] = []
  cursor = 0
  for point in points:
    if point.position > cursor:
      html_parts.append(html.escape(text[cursor : point.position]))

    if point.tag_type == TagType.START:
      colour = color_map.get(point.extraction.extraction_class, '#ffff8d')
      # Don't add lx-current-highlight here - let JavaScript handle it dynamically
      
      span_html = (
          f'<span class="lx-highlight"'
          f' data-idx="{point.span_idx}" style="background-color:{colour};">'
      )
      html_parts.append(span_html)
    else:  # point.tag_type == TagType.END
      html_parts.append('</span>')

    cursor = point.position

  if cursor < len(text):
    html_parts.append(html.escape(text[cursor:]))
  return ''.join(html_parts)


def _build_legend_html(color_map: dict[str, str]) -> str:
  """Builds legend HTML showing extraction classes and their colors."""
  if not color_map:
    return ''

  legend_items = []
  for extraction_class, colour in color_map.items():
    legend_items.append(
        '<span class="lx-label"'
        f' style="background-color:{colour};">{html.escape(extraction_class)}</span>'
    )
  return (
      '<div class="lx-legend">Extraction Classes:'
      f' {" ".join(legend_items)}</div>'
  )


def _format_attributes(attributes: dict | None) -> str:
  """Formats attributes as a single-line string."""
  if not attributes:
    return '{}'

  valid_attrs = {
      key: value
      for key, value in attributes.items()
      if value not in (None, '', 'null')
  }

  if not valid_attrs:
    return '{}'

  attrs_parts = []
  for key, value in valid_attrs.items():
    # Clean up array formatting for better readability
    if isinstance(value, list):
      value_str = ', '.join(str(v) for v in value)
    else:
      value_str = str(value)
    attrs_parts.append(
        f'<span class="lx-attr-key">{html.escape(str(key))}</span>: <span'
        f' class="lx-attr-value">{html.escape(value_str)}</span>'
    )
  return '{' + ', '.join(attrs_parts) + '}'


def _prepare_extraction_data(
    text: str,
    extractions: list[_data.Extraction],
    color_map: dict[str, str],
    context_chars: int = 150,
    show_progress: bool = False,
) -> list[dict]:
  """Prepares JavaScript data for extractions."""
  extraction_data = []
  
  # Create progress bar for processing extractions if needed
  iterator = extractions
  if show_progress and len(extractions) > 100:
    pbar = progress.create_extraction_progress_bar(
        extractions, 
        model_info="visualization", 
        disable=False
    )
    iterator = pbar
  
  for i, extraction in enumerate(iterator):
    # Assertions to inform pytype about the invariants guaranteed by _filter_valid_extractions
    assert (
        extraction.char_interval is not None
    ), 'char_interval must be non-None for valid extractions'
    assert (
        extraction.char_interval.start_pos is not None
    ), 'start_pos must be non-None for valid extractions'
    assert (
        extraction.char_interval.end_pos is not None
    ), 'end_pos must be non-None for valid extractions'

    start_pos = extraction.char_interval.start_pos
    end_pos = extraction.char_interval.end_pos

    context_start = max(0, start_pos - context_chars)
    context_end = min(len(text), end_pos + context_chars)

    before_text = text[context_start:start_pos]
    extraction_text = text[start_pos:end_pos]
    after_text = text[end_pos:context_end]

    colour = color_map.get(extraction.extraction_class, '#ffff8d')

    # Build attributes display
    attributes_html = (
        '<div><strong>class:</strong>'
        f' {html.escape(extraction.extraction_class)}</div>'
    )
    attributes_html += (
        '<div><strong>attributes:</strong>'
        f' {_format_attributes(extraction.attributes)}</div>'
    )

    extraction_data.append({
        'index': i,
        'class': extraction.extraction_class,
        'text': extraction.extraction_text,
        'color': colour,
        'startPos': start_pos,
        'endPos': end_pos,
        'beforeText': html.escape(before_text),
        'extractionText': html.escape(extraction_text),
        'afterText': html.escape(after_text),
        'attributesHtml': attributes_html,
    })

  return extraction_data


def _build_visualization_html(
    text: str,
    extractions: list[_data.Extraction],
    color_map: dict[str, str],
    animation_speed: float = 1.0,
    show_legend: bool = True,
    show_progress: bool = False,
) -> str:
  """Builds the complete visualization HTML."""
  if not extractions:
    return (
        '<div class="lx-animated-wrapper"><p>No extractions to'
        ' animate.</p></div>'
    )

  # Sort extractions by position for proper HTML nesting.
  def _extraction_sort_key(extraction):
    """Sort by position, then by span length descending for proper nesting."""
    start = extraction.char_interval.start_pos
    end = extraction.char_interval.end_pos
    span_length = end - start
    return (start, -span_length)  # longer spans first

  sorted_extractions = sorted(extractions, key=_extraction_sort_key)

  highlighted_text = _build_highlighted_text(
      text, sorted_extractions, color_map
  )
  extraction_data = _prepare_extraction_data(
      text, sorted_extractions, color_map, show_progress=show_progress
  )
  legend_html = _build_legend_html(color_map) if show_legend else ''

  js_data = json.dumps(extraction_data)

  # Prepare pos_info_str safely for pytype for the f-string below
  first_extraction = extractions[0]
  assert (
      first_extraction.char_interval
      and first_extraction.char_interval.start_pos is not None
      and first_extraction.char_interval.end_pos is not None
  ), 'first extraction must have valid char_interval with start_pos and end_pos'
  pos_info_str = f'[{first_extraction.char_interval.start_pos}-{first_extraction.char_interval.end_pos}]'

  html_content = textwrap.dedent(f"""
    <div class="lx-animated-wrapper">
      <div class="lx-attributes-panel">
        {legend_html}
        <div id="attributesContainer"></div>
      </div>
      <div class="lx-text-window" id="textWindow">
        {highlighted_text}
      </div>
      <div class="lx-controls">
        <div class="lx-button-row">
          <button class="lx-control-btn" onclick="playPause()">▶️ Play</button>
          <button class="lx-control-btn" onclick="prevExtraction()">⏮ Previous</button>
          <button class="lx-control-btn" onclick="nextExtraction()">⏭ Next</button>
        </div>
        <div class="lx-progress-container">
          <input type="range" id="progressSlider" class="lx-progress-slider" 
                 min="0" max="{len(extractions)-1}" value="0" 
                 onchange="jumpToExtraction(this.value)">
        </div>
        <div class="lx-status-text">
          <span id="entityInfo">1/{len(extractions)}</span> | 
          Pos <span id="posInfo">{pos_info_str}</span>
        </div>
      </div>
    </div>

    <script>
      (function() {{
        const extractions = {js_data};
        let currentIndex = 0;
        let isPlaying = false;
        let animationInterval = null;
        let animationSpeed = {animation_speed};

        function updateDisplay() {{
          const extraction = extractions[currentIndex];
          if (!extraction) return;

          document.getElementById('attributesContainer').innerHTML = extraction.attributesHtml;
          document.getElementById('entityInfo').textContent = (currentIndex + 1) + '/' + extractions.length;
          document.getElementById('posInfo').textContent = '[' + extraction.startPos + '-' + extraction.endPos + ']';
          document.getElementById('progressSlider').value = currentIndex;

          const playBtn = document.querySelector('.lx-control-btn');
          if (playBtn) playBtn.textContent = isPlaying ? '⏸ Pause' : '▶️ Play';

          const prevHighlight = document.querySelector('.lx-text-window .lx-current-highlight');
          if (prevHighlight) prevHighlight.classList.remove('lx-current-highlight');
          const currentSpan = document.querySelector('.lx-text-window span[data-idx="' + currentIndex + '"]');
          if (currentSpan) {{
            currentSpan.classList.add('lx-current-highlight');
            currentSpan.scrollIntoView({{block: 'center', behavior: 'smooth'}});
          }}
        }}

        function nextExtraction() {{
          currentIndex = (currentIndex + 1) % extractions.length;
          updateDisplay();
        }}

        function prevExtraction() {{
          currentIndex = (currentIndex - 1 + extractions.length) % extractions.length;
          updateDisplay();
        }}

        function jumpToExtraction(index) {{
          currentIndex = parseInt(index);
          updateDisplay();
        }}

        function playPause() {{
          if (isPlaying) {{
            clearInterval(animationInterval);
            isPlaying = false;
          }} else {{
            animationInterval = setInterval(nextExtraction, animationSpeed * 1000);
            isPlaying = true;
          }}
          updateDisplay();
        }}

        window.playPause = playPause;
        window.nextExtraction = nextExtraction;
        window.prevExtraction = prevExtraction;
        window.jumpToExtraction = jumpToExtraction;

        updateDisplay();
      }})();
    </script>""")

  return html_content


def visualize(
    data_source: _data.AnnotatedDocument | str | pathlib.Path,
    *,
    animation_speed: float = 1.0,
    show_legend: bool = True,
    gif_optimized: bool = True,
    show_progress: bool = True,
    save_html: bool = False,
) -> 'HTML | str':
  """Visualises extraction data as animated highlighted HTML.

  Args:
    data_source: Either an AnnotatedDocument or path to a JSONL file.
    animation_speed: Animation speed in seconds between extractions.
    show_legend: If ``True``, appends a colour legend mapping extraction classes
      to colours.
    gif_optimized: If ``True``, applies GIF-optimized styling with larger fonts,
      better contrast, and improved dimensions for video capture.
    save_html: If ``True``, saves the visualization as an HTML file alongside
      the input JSONL file.
    show_progress: If ``True``, shows progress bars during processing for large
      datasets (>100 extractions).

  Returns:
    An :class:`IPython.display.HTML` object if IPython is available, otherwise
    the generated HTML string.
  """
  # Load document if it's a file path
  if isinstance(data_source, (str, pathlib.Path)):
    file_path = pathlib.Path(data_source)
    if not file_path.exists():
      raise FileNotFoundError(f'JSONL file not found: {file_path}')

    if show_progress:
      # Show progress for loading documents
      file_size = file_path.stat().st_size if file_path.exists() else None
      load_pbar = progress.create_load_progress_bar(str(file_path), file_size)
      load_pbar.set_description(f"{progress.BLUE}{progress.BOLD}LangExtract{progress.RESET}: Loading for visualization")
      load_pbar.update(0)

    documents = list(_io.load_annotated_documents_jsonl(file_path))
    if not documents:
      raise ValueError(f'No documents found in JSONL file: {file_path}')

    if show_progress:
      load_pbar.close()
      progress.print_load_complete(len(documents), str(file_path))

    annotated_doc = documents[0]  # Use first document
  else:
    annotated_doc = data_source

  if not annotated_doc or annotated_doc.text is None:
    raise ValueError('annotated_doc must contain text to visualise.')

  if annotated_doc.extractions is None:
    raise ValueError('annotated_doc must contain extractions to visualise.')

  # Filter valid extractions - show ALL of them
  valid_extractions = _filter_valid_extractions(annotated_doc.extractions)

  if not valid_extractions:
    empty_html = (
        '<div class="lx-animated-wrapper"><p>No valid extractions to'
        ' animate.</p></div>'
    )
    full_html = _VISUALIZATION_CSS + empty_html
    return HTML(full_html) if HTML is not None else full_html

  # Show extraction summary if progress is enabled
  if show_progress:
    unique_classes = len({e.extraction_class for e in valid_extractions})
    progress.print_extraction_summary(
        num_extractions=len(valid_extractions),
        unique_classes=unique_classes,
        chars_processed=len(annotated_doc.text) if annotated_doc.text else 0
    )

  color_map = _assign_colors(valid_extractions)

  visualization_html = _build_visualization_html(
      annotated_doc.text,
      valid_extractions,
      color_map,
      animation_speed,
      show_legend,
      show_progress,
  )

  full_html = _VISUALIZATION_CSS + visualization_html

  # Apply GIF optimizations if requested
  if gif_optimized:
    full_html = full_html.replace(
        'class="lx-animated-wrapper"',
        'class="lx-animated-wrapper lx-gif-optimized"',
    )

  # Save HTML file if requested and data_source is a file path
  if save_html and isinstance(data_source, (str, pathlib.Path)):
    output_file = pathlib.Path(data_source)
    html_path = output_file.with_suffix('.html')
    
    if show_progress:
      save_pbar = progress.create_save_progress_bar(str(html_path))
      save_pbar.set_description(f"{progress.BLUE}{progress.BOLD}LangExtract{progress.RESET}: Saving visualization")
      save_pbar.update(0)
    
    with open(html_path, "w") as f:
        f.write(getattr(full_html, 'data', full_html))
    
    if show_progress:
      save_pbar.close()
      print(f"{progress.GREEN}✓{progress.RESET} Saved visualization to {progress.GREEN}{html_path.name}{progress.RESET}")

  if show_progress:
    progress.print_extraction_complete()

  return HTML(full_html) if HTML is not None else full_html
