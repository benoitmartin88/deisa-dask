"""The v1 refusal taxonomy: hierarchy, remedy strings and (importantly) message CONTENT.

Type-only assertions are not enough here: a refusal exists to tell the callback author WHAT was refused and HOW to get
a working path back. So every test below asserts on the text -- the offending symbol is named and the remedy string is
present -- which is also what keeps the wording from silently degrading.
"""

import pytest

from deisa.dask import runtime_errors as errors
from deisa.dask.precompute_analyzer import PrecomputeError

# Every refusal class of the v1 taxonomy (the brief's list, in the order it gives).
REFUSAL_CLASSES = [
    errors.RawFieldReadError,
    errors.MaterializationError,
    errors.NonAssociativeReductionError,
    errors.UnsupportedReductionError,
    errors.NoPrecomputableReductionError,
    errors.PrecomputeRuntimeError,
]

# One instance per factory, with the symbols/markers its message must name.
FACTORY_CASES = [
    pytest.param(
        lambda: errors.raw_field_read(".compute()", array_name="f", timestep=4),
        [".compute()", "'f'", "timestep 4"],
        id="raw_field_read",
    ),
    pytest.param(lambda: errors.numpy_materialization("the collection 'f'"), ["the collection 'f'"], id="materialize"),
    pytest.param(lambda: errors.iteration_materialization("the collection 'f'"), ["the collection 'f'"], id="iterate"),
    pytest.param(lambda: errors.ambiguous_truth("the expression 'mul'"), ["the expression 'mul'"], id="truth"),
    pytest.param(lambda: errors.non_associative("median", ".median()"), ["'median'", ".median()"], id="median"),
    pytest.param(
        lambda: errors.off_surface(".rechunk", "the attribute is not whitelisted"),
        [".rechunk", "rechunk"],
        id="off-surface",
    ),
    pytest.param(
        lambda: errors.mixed_root("the collection 'f'", "Root('f', 2)", "Root('f', 1)"),
        ["Root('f', 2)", "Root('f', 1)"],
        id="mixed-root",
    ),
    pytest.param(
        lambda: errors.cross_reduction("the reduction 'f-sum'", "it is reduced twice"),
        ["'f-sum'"],
        id="cross-reduction",
    ),
    pytest.param(lambda: errors.keepdims_unsupported(".sum()"), [".sum()", "keepdims"], id="keepdims"),
    pytest.param(lambda: errors.dtype_out_unsupported(".sum()", "dtype"), [".sum()", "dtype"], id="dtype"),
    pytest.param(lambda: errors.ddof_unsupported("std", 1, ".std()"), [".std()", "std(ddof=1)"], id="ddof"),
    pytest.param(
        lambda: errors.no_precomputable_reduction("the callback 'cb'"), ["the callback 'cb'"], id="no-reduction"
    ),
    pytest.param(
        lambda: errors.precompute_runtime_invariant("DeisaCollection", "the metadata is inconsistent"),
        ["DeisaCollection", "the metadata is inconsistent"],
        id="invariant",
    ),
]


def test_every_refusal_is_a_precompute_error():
    """The taxonomy plugs into the existing handling: ``except PrecomputeError`` still catches every refusal."""
    for kind in REFUSAL_CLASSES:
        assert issubclass(kind, errors.RuntimeRefusal)
        assert issubclass(kind, PrecomputeError)


def test_every_class_carries_a_remedy_that_names_the_escape_hatch():
    """Decision F3: the remedy target is the legacy scatter mode, so it must be reachable from every class."""
    for kind in REFUSAL_CLASSES:
        assert "precompute=False" in kind.remedy, kind.__name__
        assert len(kind.remedy) > 20, kind.__name__


def test_rewrap_keeps_the_message_and_swaps_the_remedy():
    """A raise site may point at a closer remedy without losing the cause."""
    original = errors.off_surface(".rechunk")
    rewrapped = errors.rewrap(original, "use precompute=False for now.")
    assert rewrapped.message == original.message
    assert rewrapped.remedy == "use precompute=False for now."
    assert type(rewrapped) is type(original)


@pytest.mark.parametrize("factory,markers", FACTORY_CASES)
def test_factory_message_names_the_offending_symbol_and_the_remedy(factory, markers):
    """Content, not just type: the message names what was refused and how to get a working path back."""
    refusal = factory()
    assert isinstance(refusal, errors.RuntimeRefusal)
    assert isinstance(refusal, PrecomputeError)
    text = str(refusal)
    for marker in markers:
        assert marker in text, f"{marker!r} missing from {text!r}"
    assert "precompute=False" in text
    assert "Remedy:" in text
    assert text == f"{refusal.message} Remedy: {refusal.remedy}"


@pytest.mark.parametrize("factory,markers", FACTORY_CASES)
def test_refusal_messages_stay_single_line_and_end_with_a_sentence(factory, markers):
    """A refusal is logged by the default handler, so it arrives as ONE line: no newlines, no dangling remedy."""
    refusal = factory()
    text = str(refusal)
    assert "\n" not in text
    assert refusal.message.endswith(".")
    assert refusal.remedy.endswith(".")


def test_non_associative_refusal_is_not_a_raw_field_read():
    """The deliberate reclassification: these ops used to be refused with the wrong cause and the wrong remedy."""
    assert not issubclass(errors.NonAssociativeReductionError, errors.RawFieldReadError)
    refusal = errors.non_associative("quantile", ".quantile(0.5)")
    assert isinstance(refusal, errors.NonAssociativeReductionError)
    assert "associative" in str(refusal)
    assert "ordering" in str(refusal)


def test_off_surface_remedy_spells_out_the_supported_surface():
    """A refusal is only useful if it says what WOULD have worked."""
    text = str(errors.off_surface(".fft"))
    for op_name in ("sum", "mean", "min", "max", "prod", "any", "all", "count"):
        assert op_name in text
    for operator in ("sqrt", "where", "clip"):
        assert operator in text


def test_no_precomputable_reduction_message_is_sharper_than_the_precompute_one():
    """The surviving class keeps its name but must now say WHY a callback has nothing to dispatch."""
    text = str(errors.no_precomputable_reduction("the callback 'cb'", "it only builds a pointwise chain"))
    assert "chunk-locally" in text
    assert "pointwise chain" in text


@pytest.mark.parametrize("kind", REFUSAL_CLASSES)
def test_class_docstring_states_the_cause_and_the_fix(kind):
    """Each class documents what it means, so the taxonomy is readable without reading the code."""
    docstring = kind.__doc__ or ""
    assert len(docstring) > 80
    assert kind.__name__ in dir(errors)
