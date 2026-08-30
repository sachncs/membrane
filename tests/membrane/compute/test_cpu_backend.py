"""Tests for CPU."""

from membrane.compute.cpu import CPU


class TestCPUBackend:
    """Test suite for CPU."""

    def test_available(self):
        backend = CPU()
        assert backend.available()

    def test_device_name(self):
        backend = CPU()
        assert backend.device_name() == "cpu"

    def test_prefill_returns_fragments(self):
        backend = CPU()
        tokens = list(range(512))
        frags = backend.prefill(tokens, "test-model")
        assert len(frags) > 0
        assert all(hasattr(f, "identity") for f in frags)
        assert all(f.identity.model_id == "test-model" for f in frags)

    def test_prefill_window_size(self):
        backend = CPU()
        tokens = list(range(300))
        frags = backend.prefill(tokens, "m")
        assert len(frags) == 3  # 128 + 128 + 44
