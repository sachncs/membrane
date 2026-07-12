"""Tests for GPUBackend."""

from membrane.compute.gpu_backend import GPUBackend


class TestGPUBackend:
    """Test suite for GPUBackend."""

    def test_device_name_shows_fallback_when_no_cuda(self):
        backend = GPUBackend()
        name = backend.device_name()
        if not backend.available():
            assert "fallback" in name
        else:
            assert "cuda" in name

    def test_prefill_returns_fragments(self):
        backend = GPUBackend()
        tokens = list(range(256))
        frags = backend.prefill(tokens, "m")
        assert len(frags) > 0

    def test_available_reflects_cuda(self):
        backend = GPUBackend()
        try:
            import torch

            expected = torch.cuda.is_available()
        except ImportError:
            expected = False
        assert backend.available() == expected
