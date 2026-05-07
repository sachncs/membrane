"""Tests for CPUBackend."""

from membrane.compute.cpu_backend import CPUBackend


class TestCPUBackend:
    """Test suite for CPUBackend."""

    def test_available(self):
        backend = CPUBackend()
        assert backend.available()

    def test_device_name(self):
        backend = CPUBackend()
        assert backend.device_name() == "cpu"

    def test_prefill_returns_fragments(self):
        backend = CPUBackend()
        tokens = list(range(512))
        frags = backend.prefill(tokens, "test-model")
        assert len(frags) > 0
        assert all(hasattr(f, "content_hash") for f in frags)
        assert all(f.structural_signature.model_id == "test-model" for f in frags)

    def test_prefill_window_size(self):
        backend = CPUBackend()
        tokens = list(range(300))
        frags = backend.prefill(tokens, "m")
        assert len(frags) == 3  # 128 + 128 + 44
