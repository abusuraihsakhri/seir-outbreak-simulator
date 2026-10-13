"""
Deterministic mock adapter for generic diagnostic CLI demonstrations.
"""
from typing import Dict, Any, Optional
from .base import PHIGuard


class MockLLM:
    def __init__(self, system_name: str = "Seir Outbreak Simulator"):
        self.system_name = system_name

    def invoke(self, prompt: str) -> str:
        PHIGuard.assert_no_phi(prompt)
        return (f"[{self.system_name} mock adapter]: No model call or scientific "
                f"verification was performed. Example query: '{prompt[:60]}...'.")


class LLMFactory:
    """Creates configured LLM client instances with zero-PHI protection."""

    @staticmethod
    def create(provider: str = "mock", system_name: str = "Seir Outbreak Simulator"):
        prov = str(provider).lower()
        if prov in ["mock", "deterministic", "test"]:
            return MockLLM(system_name)
        raise ValueError(
            f"Provider {provider!r} is not implemented. "
            "Only the deterministic 'mock' provider is available."
        )
