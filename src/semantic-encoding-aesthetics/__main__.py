"""
Local execution entry point.
For FastMCP Cloud, the server object is imported directly.
For local testing, this runs the event loop.
"""

from .server import mcp

if __name__ == "__main__":
    mcp.run()
