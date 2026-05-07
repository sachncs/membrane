from membrane.structural_signature import StructuralSignature


def test_create_signature():
    sig = StructuralSignature(
        model_id="kimi-linear-1t",
        layer_range=(0, 3),
        token_span=(1024, 2048),
    )
    assert sig.model_id == "kimi-linear-1t"
    assert sig.layer_range == (0, 3)
    assert sig.token_span == (1024, 2048)


def test_signature_is_hashable():
    sig = StructuralSignature("m", (0, 1), (0, 10))
    assert hash(sig) == hash(("m", (0, 1), (0, 10)))


def test_signature_equality():
    a = StructuralSignature("m", (0, 1), (0, 10))
    b = StructuralSignature("m", (0, 1), (0, 10))
    c = StructuralSignature("m", (0, 2), (0, 10))
    assert a == b
    assert a != c
