"""Attention backends, KV cache, and streaming inference.

Import the submodules directly (``inference.kv_cache``, ``inference.streaming``,
and so on). This package init stays empty so model imports cannot cycle through
a streaming re-export.
"""
