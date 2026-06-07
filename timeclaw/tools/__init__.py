"""MCP-exposed tools the TimeClaw agent can call.

Each tool is a FastMCP-decorated function. The module ``server`` exposes a
factory that builds a fresh in-memory FastMCP server with the full tool
registry; one such server is created per concurrent worker slot so that
per-task state (loaded series) stays isolated.
"""
