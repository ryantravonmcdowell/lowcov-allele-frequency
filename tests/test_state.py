"""Verifier for the low-coverage allele frequency task.

Contract checks run first (existence, header, keys, vocabulary, cell syntax,
no blank lines). Correctness checks run only after the contract fixtures
succeed. Truth is the simulated genotypes baked into this image at
/tests/truth.tsv, recorded by the generator at draw time, not by the
reference solution.
"""
import json
import os
import re
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

ARTIFACT = "/root/results/allele_frequencies.tsv"
TRUTH = "/tests/truth.tsv"
METRICS_OUT = "/logs/verifier/metrics.json"

COLUMNS = ["chrom", "pos", "locus_class", "af_P1", "af_P2"]
HEADER_LINE = "chrom\tpos\tlocus_class\taf_P1\taf_P2"
N_ROWS = 12000
CLASSES = {"single_copy", "collapsed_paralog"}
POPS = ("P1", "P2")

# gating thresholds (derivation in README Verification)
MIN_PARALOG_RECALL = 0.90
MIN_PARALOG_PRECISION = 0.94
ZERO_ABS = 0.008
RARE_CUT = 0.10
REL_RARE, FLOOR_RARE = 0.28, 0.018
REL_COMMON, FLOOR_COMMON = 0.22, 0.025
MIN_PASS = {"zero": 0.94, "rare": 0.90, "common": 0.94, "xcommon": 0.91}
EPS = 1e-9
PAR_END = 800_000

_diag = {}


def _flush_diag():
    try:
        os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
        with open(METRICS_OUT, "w") as fh:
            json.dump(_diag, fh, indent=1, sort_keys=True)
    except OSError:
        pass


@pytest.fixture(scope="session", autouse=True)
def _write_diagnostics():
    """Write the diagnostic metrics file after the session; never the reward."""
    yield
    _flush_diag()


@pytest.fixture(scope="session")
def truth():
    """Ground truth table drawn by construction by the data generator."""
    t = pd.read_csv(TRUTH, sep="\t", comment="#", dtype={"chrom": str})
    for c in ("af_P1", "af_P2"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


def _read_artifact_lines():
    """Read the artifact as text; reject missing file, blank lines, wrong
    line count or header, and any control character or malformed cell that
    pandas parsing would otherwise silently normalise away (see
    _assert_plain_text_cells)."""
    if not os.path.isfile(ARTIFACT):
        pytest.fail(f"missing output file {ARTIFACT}")
    with open(ARTIFACT, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if not text:
        pytest.fail("output file is empty")
    # reject trailing newline-only extra lines and internal blank lines
    # instruction: exactly header + 12000 data rows, no comment or blank lines
    lines = text.splitlines()
    if any(line.strip() == "" for line in lines):
        pytest.fail("output contains blank lines (instruction forbids them)")
    if text.endswith("\n\n") or text.endswith("\r\n\r\n"):
        pytest.fail("output ends with an extra blank line")
    # exactly N_ROWS + 1 lines (header + data)
    if len(lines) != N_ROWS + 1:
        pytest.fail(
            f"expected {N_ROWS + 1} non-blank lines (header + {N_ROWS} rows), "
            f"found {len(lines)}")
    # Exact first-line header (rejects "chrom " with trailing space, etc.)
    if lines[0] != HEADER_LINE:
        pytest.fail(
            f"header line must be exactly {HEADER_LINE!r}, found {lines[0]!r}")
    _assert_plain_text_cells(lines)
    return lines


@pytest.fixture(scope="session")
def raw():
    """Agent table read as strings so syntax problems stay visible.

    Enforces the no-blank-line contract before pandas parsing.
    """
    lines = _read_artifact_lines()
    from io import StringIO
    buf = StringIO("\n".join(lines) + "\n")
    return pd.read_csv(buf, sep="\t", dtype=str, keep_default_na=False)


# Integer pos: digits only, no sign, no decimal (rejects "1.0", "+1", " 1")
_POS_RE = re.compile(r"^[0-9]+$")
# Plain decimal or scientific notation, no whitespace (rejects "0 ", "0.5 ")
_FREQ_RE = re.compile(
    r"^[+-]?(?:(?:[0-9]+\.?[0-9]*)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?$"
)


def _assert_plain_text_cells(lines):
    """Validate raw line text before pandas can normalise anything away.

    The C parser behind pd.read_csv silently drops NUL bytes, so a cell
    written as "0\\x00" would otherwise reach the numeric checks as a
    clean "0" and score unchanged. Control characters are forbidden
    anywhere in the file, and every cell's syntax is checked on the raw
    tab-split string, exactly as written, not on a parser-cleaned copy.
    """
    for lineno, line in enumerate(lines, start=1):
        for ch in line:
            if (ord(ch) < 0x20 and ch != "\t") or ord(ch) == 0x7F:
                pytest.fail(
                    f"line {lineno}: control character {ch!r} is not "
                    f"allowed; the output must be plain text"
                )
        fields = line.split("\t")
        if len(fields) != len(COLUMNS):
            pytest.fail(
                f"line {lineno}: expected {len(COLUMNS)} tab-separated "
                f"fields, found {len(fields)}")
        if lineno == 1:
            continue  # exact header string already checked by the caller
        row = lineno - 2
        if not _POS_RE.match(fields[1]):
            pytest.fail(
                f"pos must be an integer string with no decimal point or "
                f"spaces; found {fields[1]!r} at row {row}")
        if fields[2] not in CLASSES:
            pytest.fail(
                f"locus_class {fields[2]!r} at row {row} is not one of "
                f"{sorted(CLASSES)}")
        if fields[2] == "single_copy":
            for name, val in (("af_P1", fields[3]), ("af_P2", fields[4])):
                if not _FREQ_RE.match(val):
                    pytest.fail(
                        f"{name}: frequency must be plain decimal or "
                        f"scientific notation with no whitespace; found "
                        f"{val!r} at row {row}")
                if not (Decimal("0") <= Decimal(val) <= Decimal("1")):
                    pytest.fail(
                        f"{name}: frequency {val!r} at row {row} is "
                        f"outside the inclusive [0, 1] range")
        else:
            for name, val in (("af_P1", fields[3]), ("af_P2", fields[4])):
                if val != "NA":
                    pytest.fail(
                        f"{name}: collapsed_paralog rows must hold the "
                        f"literal NA; found {val!r} at row {row}")


def _assert_pos_integer_strings(raw):
    """pos must be written as an integer token (instruction: pos as an integer)."""
    for i, s in enumerate(raw["pos"].astype(str)):
        if not _POS_RE.match(s):
            pytest.fail(
                f"pos must be an integer string with no decimal point or spaces; "
                f"found {s!r} at row {i}"
            )


def _assert_frequency_cells(raw):
    """single_copy: plain decimal/scientific in [0,1], no whitespace;
    collapsed_paralog: literal NA both cols; NA nowhere else."""
    single = raw["locus_class"] == "single_copy"
    para = raw["locus_class"] == "collapsed_paralog"
    for c in ("af_P1", "af_P2"):
        v_s = raw.loc[single, c]
        for i, s in zip(v_s.index, v_s.astype(str)):
            if not _FREQ_RE.match(s):
                pytest.fail(
                    f"{c}: frequency must be plain decimal or scientific "
                    f"notation with no whitespace; found {s!r} at row {i}"
                )
        for i, s in zip(v_s.index, v_s.astype(str)):
            # exact decimal comparison on the written string: a value like
            # 1.0000000000000001 rounds to 1.0 under float coercion but is
            # outside the inclusive [0, 1] range the instruction states
            if not (Decimal("0") <= Decimal(s) <= Decimal("1")):
                pytest.fail(
                    f"{c}: frequency {s!r} at row {i} is outside the "
                    f"inclusive [0, 1] range")
        num = pd.to_numeric(v_s, errors="coerce")
        if num.isna().any():
            pytest.fail(f"{c}: non-numeric or NA in single_copy rows")
        if not np.isfinite(num).all():
            pytest.fail(f"{c}: non-finite value in single_copy rows")
        if not ((num >= 0) & (num <= 1)).all():
            pytest.fail(f"{c}: value outside [0, 1] in single_copy rows")
        if not (raw.loc[para, c] == "NA").all():
            pytest.fail(f"{c}: collapsed_paralog rows must hold the literal NA")
        if (v_s == "NA").any():
            pytest.fail(f"{c}: NA appears on single_copy rows")


@pytest.fixture(scope="session")
def merged(raw, truth):
    """Truth joined to agent rows; fails hard on any contract error so
    correctness tests never see an invalid table."""
    if list(raw.columns) != COLUMNS:
        pytest.fail(f"header must be {COLUMNS}, found {list(raw.columns)}")
    if len(raw) != N_ROWS:
        pytest.fail(f"expected {N_ROWS} rows, found {len(raw)}")
    if not set(raw["locus_class"]) <= CLASSES:
        pytest.fail("locus_class outside {single_copy, collapsed_paralog}")
    _assert_pos_integer_strings(raw)
    pos = pd.to_numeric(raw["pos"], errors="coerce")
    if pos.isna().any() or (pos != pos.round()).any():
        pytest.fail("pos must be an integer in every row")
    _assert_frequency_cells(raw)
    out = raw.copy()
    out["pos"] = pos.astype(int)
    if out.duplicated(["chrom", "pos"]).any():
        pytest.fail("duplicate (chrom, pos) rows")
    for c in ("af_P1", "af_P2"):
        out[c + "_est"] = pd.to_numeric(
            out[c].where(out[c] != "NA"), errors="coerce")
    m = truth.merge(
        out[["chrom", "pos", "locus_class", "af_P1", "af_P2",
             "af_P1_est", "af_P2_est"]],
        on=["chrom", "pos"], how="outer", suffixes=("", "_out"),
        indicator=True)
    if (m["_merge"] != "both").any():
        pytest.fail("(chrom, pos) keys differ from the VCF candidate list")
    return m.rename(columns={"locus_class": "truth_class",
                             "locus_class_out": "out_class"})


# ---------------------------------------------------------------- contract
def test_output_file_exists():
    """The declared artifact path is a regular file."""
    assert os.path.isfile(ARTIFACT), f"missing output file {ARTIFACT}"


def test_no_blank_lines():
    """The output has no blank lines and exactly header + 12,000 data rows."""
    _read_artifact_lines()


def test_header_is_exact(raw):
    """The first line is the exact documented header; parsed names match too."""
    lines = _read_artifact_lines()
    assert lines[0] == HEADER_LINE, (
        f"header line must be exactly {HEADER_LINE!r}, found {lines[0]!r}")
    assert list(raw.columns) == COLUMNS, (
        f"header must be {COLUMNS}, found {list(raw.columns)}")


def test_row_count_and_keys(raw, truth):
    """One row per candidate site: 12,000 rows, unique integer (chrom, pos)
    keys equal to the candidate list. pos must be an integer string."""
    assert len(raw) == N_ROWS, f"expected {N_ROWS} rows, found {len(raw)}"
    _assert_pos_integer_strings(raw)
    pos = pd.to_numeric(raw["pos"], errors="coerce")
    assert not pos.isna().any(), "pos has non-numeric entries"
    keys = set(zip(raw["chrom"], pos.astype(int)))
    assert len(keys) == N_ROWS, "duplicate (chrom, pos) rows"
    expect = set(zip(truth["chrom"], truth["pos"].astype(int)))
    assert keys == expect, "(chrom, pos) keys differ from the VCF candidate list"


def test_locus_class_vocabulary(raw):
    """locus_class takes only the two documented values."""
    bad = sorted(set(raw["locus_class"]) - CLASSES)
    assert not bad, f"unknown locus_class values {bad}"


def test_frequency_cells_wellformed(raw):
    """single_copy rows carry finite frequencies in [0, 1] in both columns;
    collapsed_paralog rows carry the literal NA in both; NA appears nowhere
    else."""
    _assert_frequency_cells(raw)


# ----------------------------------------------------------- correctness
def _paralog_scores(m):
    pred = m["out_class"] == "collapsed_paralog"
    true = m["truth_class"] == "collapsed_paralog"
    tp = int((pred & true).sum())
    recall = tp / max(int(true.sum()), 1)
    precision = tp / max(int(pred.sum()), 1)
    return recall, precision


def test_paralog_recall(merged):
    """Enough of the truly collapsed paralog sites carry the
    collapsed_paralog label."""
    recall, precision = _paralog_scores(merged)
    _diag["paralog_recall"] = recall
    _diag["paralog_precision"] = precision
    assert recall >= MIN_PARALOG_RECALL, (
        f"collapsed_paralog recall {recall:.3f} < {MIN_PARALOG_RECALL}")


def test_paralog_precision(merged):
    """Enough of the sites labelled collapsed_paralog are truly paralogs."""
    recall, precision = _paralog_scores(merged)
    _diag["paralog_recall"] = recall
    _diag["paralog_precision"] = precision
    assert precision >= MIN_PARALOG_PRECISION, (
        f"collapsed_paralog precision {precision:.3f} < {MIN_PARALOG_PRECISION}")


def _pass_fraction(m, pop, group):
    """Share of truly single_copy sites in a MAF stratum whose estimate is
    within tolerance. Mis-labelled paralogs count as misses."""
    est = m[f"af_{pop}_est"]
    t = m[f"af_{pop}"]
    single = m["truth_class"] == "single_copy"
    # agent must also label single_copy for the estimate to count
    ok_label = m["out_class"] == "single_copy"
    mf = np.minimum(t, 1.0 - t)
    nonpar_x = (m["chrom"] == "chrX") & (m["pos"] > PAR_END)
    if group == "zero":
        sel = single & ((t <= EPS) | (t >= 1.0 - EPS))
        tol = ZERO_ABS
        err = (est - t).abs()
        ok = (err <= tol + EPS) & est.notna() & ok_label
    elif group == "rare":
        sel = single & (mf > 0) & (mf < RARE_CUT)
        tol = np.maximum(REL_RARE * mf, FLOOR_RARE)
        err = (est - t).abs()
        ok = (err <= tol + EPS) & est.notna() & ok_label
    elif group == "common":
        sel = single & (mf >= RARE_CUT)
        tol = np.maximum(REL_COMMON * mf, FLOOR_COMMON)
        err = (est - t).abs()
        ok = (err <= tol + EPS) & est.notna() & ok_label
    else:
        sel = single & nonpar_x & (mf >= RARE_CUT)
        tol = np.maximum(REL_COMMON * mf, FLOOR_COMMON)
        err = (est - t).abs()
        ok = (err <= tol + EPS) & est.notna() & ok_label
    n = int(sel.sum())
    assert n > 0, f"no truth sites in group {group}"
    return float(ok[sel].mean()), n


@pytest.mark.parametrize("pop", POPS)
def test_zero_frequency_sites(merged, pop):
    """At sites where the population carries only one allele, estimates sit
    within the absolute bound of the true 0 or 1 for enough sites."""
    frac, n = _pass_fraction(merged, pop, "zero")
    _diag[f"{pop}_zero"] = frac
    assert frac >= MIN_PASS["zero"], (
        f"{pop} monomorphic-in-population sites: {frac:.3f} within bound "
        f"(n={n}) < {MIN_PASS['zero']}")


@pytest.mark.parametrize("pop", POPS)
def test_rare_frequency_sites(merged, pop):
    """At sites with true minor allele frequency in (0, 0.10), enough
    estimates fall inside the relative tolerance with its absolute floor."""
    frac, n = _pass_fraction(merged, pop, "rare")
    _diag[f"{pop}_rare"] = frac
    assert frac >= MIN_PASS["rare"], (
        f"{pop} rare sites: {frac:.3f} within tolerance (n={n}) < "
        f"{MIN_PASS['rare']}")


@pytest.mark.parametrize("pop", POPS)
def test_common_frequency_sites(merged, pop):
    """At sites with true minor allele frequency of 0.10 or more, enough
    estimates fall inside the relative tolerance with its absolute floor."""
    frac, n = _pass_fraction(merged, pop, "common")
    _diag[f"{pop}_common"] = frac
    assert frac >= MIN_PASS["common"], (
        f"{pop} common sites: {frac:.3f} within tolerance (n={n}) < "
        f"{MIN_PASS['common']}")


@pytest.mark.parametrize("pop", POPS)
def test_common_sites_nonpar_chrx(merged, pop):
    """The common-site check restricted to non-pseudo-autosomal chrX, where
    male haplotype counting differs from the autosomes."""
    frac, n = _pass_fraction(merged, pop, "xcommon")
    _diag[f"{pop}_xcommon"] = frac
    assert frac >= MIN_PASS["xcommon"], (
        f"{pop} non-PAR chrX common sites: {frac:.3f} within tolerance "
        f"(n={n}) < {MIN_PASS['xcommon']}")
