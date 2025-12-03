"""Semantic Encoding Aesthetics MCP Server

Multi-modal generative art with sentiment-driven parameters.
Supports visual primitives (geometric, image, audio waveforms) and
audio synthesis (Morse, Braille, Dot Matrix).
"""

from .server import mcp

__version__ = "0.2.0"
__all__ = ["mcp"]
