"""API generation must retain public contracts and fail on incomplete output."""

import importlib.util
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "api_reference", REPO / "docs/website/scripts/build_api_reference.py"
)
api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api)


def test_public_profile_methods_and_cluster_properties(tmp_path):
    api._emit_module_page("progenax.profiles", tmp_path / "profiles.md", REPO / "src")
    text = (tmp_path / "profiles.md").read_text()
    for member in ("PlummerProfile.density", "KingProfile.from_W0_rc"):
        assert f"### `{member}`" in text
    api._emit_module_page("progenax.cluster", tmp_path / "cluster.md", REPO / "src")
    assert (
        "### `MultiComponentCluster.rescale_j`" in (tmp_path / "cluster.md").read_text()
    )


def test_docstring_markup_preserves_warning_and_example():
    def example():
        """See :func:`private_helper` and :math:`r^2`.

        .. warning::
            Keep the physical limitation.

        Examples:
            >>> for i in range(2):
            ...     print(i)
            0
            1
        """

    text = api._format_docstring(example)
    assert ":func:" not in text and ".. warning::" not in text
    assert ":::{warning}" in text and "Keep the physical limitation." in text
    assert "$r^2$" in text and "`private_helper`" in text
    assert "```pycon\n>>> for i in range(2):\n...     print(i)\n0\n1\n```" in text


def test_numpy_sections_keep_types_and_paragraphs():
    def example():
        """A contract.

        Parameters
        ----------
        x : float
            Radius in pc.

            Must be positive.

        Returns
        -------
        float
            A squared radius in pc^2.
        """

    text = api._format_docstring(example)
    assert "----------" not in text
    for content in (
        "**Parameters**",
        "float",
        "Radius in pc.",
        "Must be positive.",
        "pc^2",
    ):
        assert content in text


def test_generation_failure_returns_nonzero_with_owner(monkeypatch, tmp_path, capsys):
    def broken(*args, **kwargs):
        raise ValueError("PlummerProfile.density: broken contract")

    monkeypatch.setattr(api, "_emit_module_page", broken)
    assert api.main(["--out", str(tmp_path), "--package-root", str(REPO / "src")]) == 1
    assert "PlummerProfile.density" in capsys.readouterr().err


@pytest.fixture(scope="module")
def generated_reference(tmp_path_factory):
    directory = tmp_path_factory.mktemp("complete-api")
    assert api.main(["--out", str(directory)]) == 0
    return directory


def test_inherited_methods_link_to_contract_owner(generated_reference):
    text = (generated_reference / "imf.md").read_text()
    assert "### `Maschberger.sample_fixed_n`" in text
    assert (
        "Inherited from [`BaseIMF.sample_fixed_n`](imf.md#api-imf-baseimf-sample_fixed_n)"
        in text
    )
    assert "kroupa(cls" not in text
    assert "### `PowerLawIMF.kroupa`" in text


def test_all_generated_public_callables_have_source_docstrings(generated_reference):
    missing = []
    for module in api.PUBLIC_MODULES:
        for name, obj in api._collect_public_symbols(module):
            if api._classify(obj) in ("function", "class", "protocol"):
                if not inspect.getdoc(obj):
                    missing.append(f"{module}.{name}")
            if inspect.isclass(obj):
                for member, method, _kind, _owner in api._collect_class_members(obj):
                    if not inspect.getdoc(method):
                        missing.append(f"{module}.{name}.{member}")
    assert not missing, "Missing public contracts: " + ", ".join(missing)
    for page in generated_reference.glob("*.md"):
        assert "*(no docstring)*" not in page.read_text(), page.name


def test_profile_name_is_a_type_alias(generated_reference):
    from progenax.profiles import ProfileName

    assert api._classify(ProfileName) == "type alias"
    text = (generated_reference / "profiles.md").read_text()
    assert "ProfileName(*args" not in text
    assert "ProfileName = Literal['plummer', 'king', 'eff']" in text
    index = (generated_reference / "full-symbol-index.md").read_text()
    assert (
        "[`ProfileName`](profiles.md#api-profiles-profilename) | type alias |" in index
    )


def test_index_includes_missing_exports_protocols_and_methods(generated_reference):
    text = (generated_reference / "full-symbol-index.md").read_text()
    for name in (
        "DEFAULT_UNITS",
        "zams_radius",
        "zams_luminosity",
        "zams_effective_temperature",
        "zams_surface_gravity",
        "inverse_zams_luminosity",
        "SpatialProfile.sample_positions",
        "MultiComponentCluster.from_imf",
        "MultiComponentCluster.rescale_j",
        "KeplerElements.to_state",
    ):
        assert f"[`{name}`]" in text
    for path in generated_reference.glob("*.md"):
        body = path.read_text()
        assert not api._ROLE.search(body), path.name
        assert not api._DIRECTIVE.search(body), path.name


def test_check_detects_staleness_without_writing(generated_reference, capsys):
    path = generated_reference / "stellar.md"
    original = path.read_bytes()
    assert api.main(["--out", str(generated_reference), "--check"]) == 0
    path.write_text("stale\n")
    try:
        assert api.main(["--out", str(generated_reference), "--check"]) == 1
        assert "stellar.md" in capsys.readouterr().err
        assert path.read_text() == "stale\n"
    finally:
        path.write_bytes(original)
    assert api.main(["--out", str(generated_reference), "--check"]) == 0


def test_roles_resolve_public_contracts_and_preserve_code():
    text = api._convert_markup(
        "See :meth:`sample` and :func:`_internal`.\n\n"
        "```python\n# :func:`literal_example`\n```",
        {"progenax.imf.BaseIMF.sample": "imf.md#api-imf-baseimf-sample"},
        "progenax.imf.BaseIMF",
    )
    assert "[`sample`](imf.md#api-imf-baseimf-sample)" in text
    assert "`_internal`" in text
    assert "# :func:`literal_example`" in text


def test_google_preamble_and_nested_content_are_retained():
    def example():
        """Contract.

        Args:
            Use consistent units.
            radius: Radius in pc.

        Returns:
            - A radius.
            - A mass.

        .. math::

            r^2 = x^2 + y^2
        """

    text = api._format_docstring(example)
    assert "Use consistent units." in text
    assert "- A radius.\n- A mass." in text
    assert "$$\nr^2 = x^2 + y^2\n$$" in text


def test_missing_export_is_an_error(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        api.importlib,
        "import_module",
        lambda name: SimpleNamespace(__all__=["missing"]),
    )
    with pytest.raises(ValueError, match="progenax.example.missing"):
        list(api._collect_public_symbols("progenax.example"))


def test_descriptors_are_documented_without_evaluating_properties(tmp_path):
    class Example:
        __module__ = "progenax.profiles"

        @staticmethod
        def scale(radius):
            """Return the input radius."""
            return radius

        @classmethod
        def construct(cls, radius):
            """Construct from a radius."""
            return cls()

        @property
        def radius(self):
            """The stored radius in profile length units."""
            raise AssertionError("generation must not evaluate properties")

    catalog = {"progenax.profiles": [("Example", Example)]}
    page = tmp_path / "profiles.md"
    api._emit_module_page("progenax.profiles", page, REPO / "src", catalog=catalog)
    text = page.read_text()
    assert "*staticmethod*" in text and "scale(radius)" in text
    assert "construct(radius)" in text and "construct(cls" not in text
    assert "### `Example.radius`" in text and "stored radius" in text


def test_native_myst_and_python_fences_are_preserved():
    def example():
        """Contract.

        Examples:
            ```pycon
            >>> print(1)
            1
            ```

        :::{note}
        Use explicit units.
        :::
        """

    text = api._format_docstring(example)
    assert text.count("```pycon") == 1
    assert ":::{note}\nUse explicit units.\n:::" in text
