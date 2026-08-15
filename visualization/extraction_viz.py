"""
Interactive visualization for extractions.

Generates HTML visualizations showing extractions highlighted
in their source text for debugging and verification.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ctxforge.protocols.extractor import ExtractionCandidate
from ctxforge.protocols.graph import GraphEdge, GraphNode

# Color palette for different extraction types
_PALETTE = [
    '#D2E3FC',  # Light Blue
    '#C8E6C9',  # Light Green
    '#FEF0C3',  # Light Yellow
    '#F9DEDC',  # Light Red
    '#FFDDBE',  # Light Orange
    '#EADDFF',  # Light Purple
    '#C4E9E4',  # Light Teal
    '#FCE4EC',  # Light Pink
]


@dataclass
class HighlightedSpan:
    """A span to highlight in text."""
    
    start_pos: int
    end_pos: int
    label: str
    color: str
    tooltip: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos


def visualize_memory_extractions(
    source_text: str,
    candidates: List[ExtractionCandidate],
    title: str = "Memory Extractions",
) -> str:
    """
    Generate HTML visualization for memory extractions.
    
    Args:
        source_text: The original text
        candidates: List of extraction candidates
        title: Title for the visualization
        
    Returns:
        HTML string with interactive visualization
    """
    # Build highlights
    highlights: List[HighlightedSpan] = []
    type_colors: Dict[str, str] = {}
    
    for c in candidates:
        if c.source_span is None:
            continue
        
        # source_span is now CharSpan
        start_pos = c.source_span.start_pos
        end_pos = c.source_span.end_pos
        
        # Assign color by type
        type_str = c.memory_type.value
        if type_str not in type_colors:
            type_colors[type_str] = _PALETTE[len(type_colors) % len(_PALETTE)]
        
        tooltip = f"{type_str}: {c.content}\nConfidence: {c.confidence:.2f}"
        if c.alignment_status:
            # alignment_status is now AlignmentStatus enum
            tooltip += f"\nAlignment: {c.alignment_status.value}"
        
        highlights.append(HighlightedSpan(
            start_pos=start_pos,
            end_pos=end_pos,
            label=type_str,
            color=type_colors[type_str],
            tooltip=tooltip,
            data={
                "content": c.content,
                "type": type_str,
                "confidence": c.confidence,
                "tags": c.tags,
            },
        ))
    
    return _build_html(source_text, highlights, title, type_colors)


def visualize_graph_extractions(
    source_text: str,
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    episode_id: str,
    title: str = "Graph Extractions",
) -> str:
    """
    Generate HTML visualization for graph extractions.
    
    Args:
        source_text: The original text
        nodes: List of graph nodes
        edges: List of graph edges
        episode_id: ID of the source episode
        title: Title for the visualization
        
    Returns:
        HTML string with interactive visualization
    """
    highlights: List[HighlightedSpan] = []
    type_colors: Dict[str, str] = {}
    
    # Add node highlights
    for node in nodes:
        if episode_id not in node.source_spans:
            continue
        
        # source_spans[episode_id] is now CharSpan
        char_span = node.source_spans[episode_id]
        start_pos = char_span.start_pos
        end_pos = char_span.end_pos
        
        label = node.labels[0] if node.labels else "Entity"
        if label not in type_colors:
            type_colors[label] = _PALETTE[len(type_colors) % len(_PALETTE)]
        
        tooltip = f"{label}: {node.name}"
        if node.summary:
            tooltip += f"\n{node.summary}"
        
        highlights.append(HighlightedSpan(
            start_pos=start_pos,
            end_pos=end_pos,
            label=label,
            color=type_colors[label],
            tooltip=tooltip,
            data={
                "node_id": node.node_id,
                "name": node.name,
                "labels": node.labels,
            },
        ))
    
    # Add edge fact highlights
    for edge in edges:
        if episode_id not in edge.source_spans or not edge.fact:
            continue
        
        # source_spans[episode_id] is now CharSpan
        char_span = edge.source_spans[episode_id]
        start_pos = char_span.start_pos
        end_pos = char_span.end_pos
        
        label = f"Edge:{edge.edge_type}"
        if label not in type_colors:
            type_colors[label] = _PALETTE[len(type_colors) % len(_PALETTE)]
        
        tooltip = f"Relationship: {edge.edge_type}\nFact: {edge.fact}"
        
        highlights.append(HighlightedSpan(
            start_pos=start_pos,
            end_pos=end_pos,
            label=edge.edge_type,
            color=type_colors[label],
            tooltip=tooltip,
            data={
                "edge_id": edge.edge_id,
                "edge_type": edge.edge_type,
                "fact": edge.fact,
            },
        ))
    
    return _build_html(source_text, highlights, title, type_colors)


def _build_html(
    text: str,
    highlights: List[HighlightedSpan],
    title: str,
    color_map: Dict[str, str],
) -> str:
    """Build the complete HTML visualization."""
    
    # Sort highlights by position (start position, then longer spans first)
    sorted_highlights = sorted(highlights, key=lambda h: (h.start_pos, -h.length))
    
    # Build highlighted text (non-overlapping spans only)
    html_parts: List[str] = []
    cursor = 0
    
    for h in sorted_highlights:
        # Skip if this span overlaps with already processed text
        if h.start_pos < cursor:
            continue
        
        # Add text before highlight
        if h.start_pos > cursor:
            html_parts.append(html.escape(text[cursor:h.start_pos]))
        
        # Add highlighted span
        span_text = text[h.start_pos:h.end_pos]
        tooltip_html = html.escape(h.tooltip).replace('\n', '<br>')
        
        html_parts.append(
            f'<span class="highlight" '
            f'style="background-color:{h.color};" '
            f'data-info="{html.escape(json.dumps(h.data))}">'
            f'<span class="tooltip">{tooltip_html}</span>'
            f'{html.escape(span_text)}'
            f'</span>'
        )
        
        cursor = h.end_pos
    
    # Add remaining text
    if cursor < len(text):
        html_parts.append(html.escape(text[cursor:]))
    
    highlighted_text = ''.join(html_parts)
    
    # Build legend
    legend_items: List[str] = []
    for label, color in color_map.items():
        legend_items.append(
            f'<span class="legend-item" style="background-color:{color};">{html.escape(label)}</span>'
        )
    legend_html = ' '.join(legend_items)
    
    # Build full HTML
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{ color: #333; }}
        .legend {{
            margin-bottom: 20px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 8px;
        }}
        .legend-item {{
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .text-container {{
            font-family: monospace;
            white-space: pre-wrap;
            padding: 20px;
            background: #fafafa;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .highlight {{
            position: relative;
            border-radius: 3px;
            padding: 1px 2px;
            cursor: pointer;
        }}
        .highlight .tooltip {{
            visibility: hidden;
            opacity: 0;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 1000;
            transition: opacity 0.2s;
        }}
        .highlight:hover .tooltip {{
            visibility: visible;
            opacity: 1;
        }}
    </style>
</head>
<body>
    <h1>{html.escape(title)}</h1>
    <div class="legend">
        <strong>Legend:</strong> {legend_html}
    </div>
    <div class="text-container">{highlighted_text}</div>
    <script>
        // Click to show full info
        document.querySelectorAll('.highlight').forEach(el => {{
            el.addEventListener('click', () => {{
                const info = JSON.parse(el.dataset.info);
                console.log('Extraction info:', info);
            }});
        }});
    </script>
</body>
</html>'''


def save_visualization(html_content: str, filepath: str) -> Path:
    """
    Save HTML visualization to a file.
    
    Args:
        html_content: The HTML content to save
        filepath: Path where to save the file
        
    Returns:
        Path to the saved file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_content, encoding="utf-8")
    return path

