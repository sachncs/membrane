from membrane.signature import Signature


def test_create_signature():
    sig = Signature(
        model_id="kimi-linear-1t",
        layer_range=(0, 3),
        token_span=(1024, 2048),
    )
    assert sig.model_id == "kimi-linear-1t"
    assert sig.layer_range == (0, 3)
    assert sig.token_span == (1024, 2048)


def test_signature_is_hashable():
    sig = Signature("m", (0, 1), (0, 10))
    assert hash(sig) == hash(("m", (0, 1), (0, 10)))


def test_signature_equality():
    a = Signature("m", (0, 1), (0, 10))
    b = Signature("m", (0, 1), (0, 10))
    c = Signature("m", (0, 2), (0, 10))
    assert a == b
    assert a != c
