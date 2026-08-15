"""
Pytest configuration and fixtures for ctxforge tests.
"""

import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Alternative: use pytest-asyncio's event_loop_policy
pytest_plugins = ('pytest_asyncio',)

