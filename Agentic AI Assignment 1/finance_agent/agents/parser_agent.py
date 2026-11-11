"""
finance_agent/agents/parser_agent.py
-------------------------------------
Parser Agent — Step 1 (fully implemented)
==========================================
Responsibility:
  • Accept a CSV or PDF bank/UPI statement uploaded via Streamlit.
  • Return a clean, normalised Pandas DataFrame with columns:
      date | description | amount | transaction_type | raw_text

Design decisions:
  • CSV parsing uses Pandas with smart column-name detection so it works
    with exports from most Indian banks (SBI, HDFC, ICICI, Paytm, PhonePe).
  • PDF parsing uses pdfplumber to extract tables; falls back to raw-text
    heuristics when no table is found.
  • All cleaning (dtype coercion, duplicate removal, missing-value handling)
    is centralised in _clean_dataframe().
  • The agent exposes a single public method: `parse(file) -> ParseResult`.
"""

from __future__ import annotations

import io
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data contract returned by the agent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    """Structured return value from ParserAgent.parse()."""

    success: bool
    df: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    source_format: str = "unknown"   # "csv" | "pdf"
    row_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Column-name aliases used by various Indian banks
# ─────────────────────────────────────────────────────────────────────────────

# Maps normalised column name → possible raw header strings (lowercase)
_COLUMN_ALIASES: dict[str, list[str]] = {
    "date": [
        "date", "transaction date", "txn date", "value date",
        "posting date", "trans date", "dt",
    ],
    "description": [
        "description", "narration", "particulars", "details",
        "remarks", "transaction details", "txn details", "note",
    ],
    "amount": [
        "amount", "transaction amount", "txn amount", "debit/credit",
        "withdrawal amt (inr)", "deposit amt (inr)", "net amount",
    ],
    "debit": [
        "debit", "withdrawal", "withdrawal amount", "dr",
        "debit amount", "debits",
    ],
    "credit": [
        "credit", "deposit", "deposit amount", "cr",
        "credit amount", "credits",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Parser Agent
# ─────────────────────────────────────────────────────────────────────────────

class ParserAgent:
    """
    Reads an uploaded bank/UPI statement and returns a clean DataFrame.

    Usage::

        agent = ParserAgent()
        result = agent.parse(uploaded_file)   # Streamlit UploadedFile object
        if result.success:
            df = result.df
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, file) -> ParseResult:
        """
        Entry point: detect file type, delegate to the right parser,
        then clean and return the result.

        Parameters
        ----------
        file : Streamlit UploadedFile | file-like object | pathlib.Path | str
            The uploaded bank statement.

        Returns
        -------
        ParseResult
        """
        try:
            filename, file_bytes = self._read_file(file)
        except Exception as exc:
            return ParseResult(success=False, error=f"Could not read file: {exc}")

        ext = filename.rsplit(".", 1)[-1].lower()

        if ext == "csv":
            return self._parse_csv(file_bytes)
        elif ext == "pdf":
            return self._parse_pdf(file_bytes)
        else:
            return ParseResult(
                success=False,
                error=f"Unsupported file type '.{ext}'. Upload a CSV or PDF.",
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _read_file(file) -> tuple[str, bytes]:
        """Normalise various file-like inputs to (filename, bytes)."""
        # Streamlit UploadedFile
        if hasattr(file, "name") and hasattr(file, "read"):
            return file.name, file.read()
        # Path-like
        from pathlib import Path
        if isinstance(file, (str, Path)):
            p = Path(file)
            return p.name, p.read_bytes()
        # Raw bytes/BytesIO
        if isinstance(file, (bytes, bytearray)):
            return "statement.csv", bytes(file)
        if isinstance(file, io.IOBase):
            return "statement.csv", file.read()
        raise TypeError(f"Unsupported file type: {type(file)}")

    # ── CSV Parser ────────────────────────────────────────────────────────────

    def _parse_csv(self, raw: bytes) -> ParseResult:
        """
        Parse a CSV bank statement into a normalised DataFrame.

        Strategy:
          1. Try UTF-8, then latin-1 encoding.
          2. Skip up to 10 leading metadata rows that banks often include.
          3. Detect and rename columns using _COLUMN_ALIASES.
          4. If separate debit/credit columns exist, merge them into 'amount'.
          5. Run _clean_dataframe().
        """
        warnings: list[str] = []

        # ── 1. Decode raw bytes ───────────────────────────────────────────
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return ParseResult(
                success=False, error="Could not decode CSV (tried utf-8, latin-1)."
            )

        # ── 2. Skip leading metadata rows ─────────────────────────────────
        df_raw: Optional[pd.DataFrame] = None
        lines = text.splitlines()
        for skip in range(0, min(10, len(lines))):
            try:
                candidate = pd.read_csv(
                    io.StringIO("\n".join(lines[skip:])),
                    dtype=str,
                    skip_blank_lines=True,
                )
                # Accept if we have at least 2 columns and 1 data row
                if candidate.shape[1] >= 2 and candidate.shape[0] >= 1:
                    df_raw = candidate
                    if skip > 0:
                        warnings.append(f"Skipped {skip} header/metadata row(s).")
                    break
            except Exception:
                continue

        if df_raw is None or df_raw.empty:
            return ParseResult(success=False, error="CSV appears empty or unreadable.")

        # ── 3. Rename columns ─────────────────────────────────────────────
        df_raw = self._rename_columns(df_raw, warnings)

        # ── 4. Merge debit/credit into amount ─────────────────────────────
        df_raw = self._merge_debit_credit(df_raw, warnings)

        # ── 5. Validate required columns ──────────────────────────────────
        required = {"date", "description", "amount"}
        missing = required - set(df_raw.columns)
        if missing:
            return ParseResult(
                success=False,
                error=(
                    f"Could not find required columns: {missing}. "
                    f"Found: {list(df_raw.columns)}"
                ),
                warnings=warnings,
            )

        # ── 6. Clean ──────────────────────────────────────────────────────
        df_clean = self._clean_dataframe(df_raw, warnings)

        return ParseResult(
            success=True,
            df=df_clean,
            warnings=warnings,
            source_format="csv",
            row_count=len(df_clean),
        )

    # ── PDF Parser ────────────────────────────────────────────────────────────

    def _parse_pdf(self, raw: bytes) -> ParseResult:
        """
        Extract transaction table from a PDF bank statement.

        Strategy:
          1. Use pdfplumber to extract tables from every page.
          2. Concatenate all extracted tables.
          3. Use the first row as the header.
          4. Fall back to raw text heuristics if no table found.
          5. Delegate to the same column normalisation + cleaning pipeline.
        """
        warnings: list[str] = []

        try:
            import pdfplumber
        except ImportError:
            return ParseResult(
                success=False,
                error="pdfplumber is not installed. Run: pip install pdfplumber",
            )

        try:
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                all_tables: list[list[list]] = []
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    for tbl in tables:
                        if tbl:
                            all_tables.append(tbl)
                            logger.debug(
                                "Page %d: extracted table with %d rows", page_num, len(tbl)
                            )

            if all_tables:
                return self._pdf_tables_to_df(all_tables, warnings)
            else:
                # Fallback: parse raw text lines
                warnings.append(
                    "No structured table found in PDF; attempting text-based parsing."
                )
                return self._pdf_text_fallback(raw, warnings)

        except Exception as exc:
            return ParseResult(success=False, error=f"PDF parsing failed: {exc}")

    def _pdf_tables_to_df(
        self, tables: list[list[list]], warnings: list[str]
    ) -> ParseResult:
        """Convert pdfplumber table data into a normalised DataFrame."""
        rows: list[list] = []
        header: list[str] = []

        for tbl in tables:
            if not tbl:
                continue
            if not header:
                # First non-empty row of the first table = header
                header = [str(c).strip() if c else f"col_{i}"
                          for i, c in enumerate(tbl[0])]
                rows.extend(tbl[1:])
            else:
                # Subsequent tables: skip header row if it matches
                start = 1 if tbl[0] == header else 0  # type: ignore[comparison-overlap]
                rows.extend(tbl[start:])

        if not header or not rows:
            return ParseResult(success=False, error="PDF tables were empty after extraction.")

        df_raw = pd.DataFrame(rows, columns=header, dtype=str)
        df_raw = self._rename_columns(df_raw, warnings)
        df_raw = self._merge_debit_credit(df_raw, warnings)

        required = {"date", "description", "amount"}
        missing = required - set(df_raw.columns)
        if missing:
            return ParseResult(
                success=False,
                error=(
                    f"Required columns not found in PDF: {missing}. "
                    f"Found: {list(df_raw.columns)}"
                ),
                warnings=warnings,
            )

        df_clean = self._clean_dataframe(df_raw, warnings)
        return ParseResult(
            success=True, df=df_clean, warnings=warnings,
            source_format="pdf", row_count=len(df_clean),
        )

    def _pdf_text_fallback(self, raw: bytes, warnings: list[str]) -> ParseResult:
        """
        Last-resort parser: extract lines from PDF text and use regex to
        identify date + amount patterns typical of bank statements.

        Produces minimal columns: date, description, amount.
        """
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                full_text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except Exception as exc:
            return ParseResult(success=False, error=f"Text extraction failed: {exc}")

        # Pattern: DD/MM/YYYY or DD-MM-YYYY followed by text and a number
        pattern = re.compile(
            r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})"   # date
            r"\s+(.+?)\s+"                            # description (non-greedy)
            r"([\d,]+\.?\d*)",                        # amount
            re.MULTILINE,
        )

        records = []
        for m in pattern.finditer(full_text):
            records.append(
                {"date": m.group(1), "description": m.group(2), "amount": m.group(3)}
            )

        if not records:
            return ParseResult(
                success=False,
                error="Could not extract any transactions from PDF text.",
                warnings=warnings,
            )

        warnings.append(
            f"Text-based fallback extracted {len(records)} transactions. "
            "Accuracy may be lower than with a structured table."
        )
        df_raw = pd.DataFrame(records)
        df_clean = self._clean_dataframe(df_raw, warnings)
        return ParseResult(
            success=True, df=df_clean, warnings=warnings,
            source_format="pdf", row_count=len(df_clean),
        )

    # ── Column normalisation ──────────────────────────────────────────────────

    @staticmethod
    def _rename_columns(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
        """
        Rename raw headers to our canonical set using _COLUMN_ALIASES.
        Unrecognised columns are kept as-is (they might contain balance info, etc.).
        """
        raw_cols = {c: c.lower().strip() for c in df.columns}
        rename_map: dict[str, str] = {}

        for canonical, aliases in _COLUMN_ALIASES.items():
            for raw_col, raw_lower in raw_cols.items():
                if raw_lower in aliases and canonical not in rename_map.values():
                    rename_map[raw_col] = canonical
                    break

        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug("Renamed columns: %s", rename_map)

        return df

    @staticmethod
    def _merge_debit_credit(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
        """
        If the statement has separate 'debit' and 'credit' columns but no
        'amount' column, combine them:
          • debits  → negative amounts
          • credits → positive amounts
        """
        has_amount = "amount" in df.columns
        has_debit = "debit" in df.columns
        has_credit = "credit" in df.columns

        if has_amount or not (has_debit or has_credit):
            return df  # nothing to do

        def _to_float(series: pd.Series) -> pd.Series:
            return (
                series.astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[^\d.\-]", "", regex=True)
                .replace("", np.nan)
                .astype(float)
            )

        debit_vals = _to_float(df["debit"]) if has_debit else pd.Series(0.0, index=df.index)
        credit_vals = _to_float(df["credit"]) if has_credit else pd.Series(0.0, index=df.index)

        # Credits are positive, debits are negative
        df["amount"] = credit_vals.fillna(0) - debit_vals.fillna(0)
        df["transaction_type"] = np.where(
            df["amount"] >= 0, "credit", "debit"
        )

        # Drop originals to avoid confusion
        df = df.drop(columns=[c for c in ["debit", "credit"] if c in df.columns])
        warnings.append("Merged separate debit/credit columns into unified 'amount'.")
        return df

    # ── Data cleaning ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
        """
        Standardise dtypes, remove junk rows, and add derived columns.

        Steps:
          1. Coerce 'date' → datetime (try multiple formats).
          2. Coerce 'amount' → float (strip currency symbols, commas).
          3. Add 'transaction_type' if not already present.
          4. Add 'raw_text' column (original description, unchanged).
          5. Drop fully duplicate rows.
          6. Drop rows where date or amount is null.
          7. Sort by date ascending.
          8. Reset index.
        """
        # ── 1. Date coercion ──────────────────────────────────────────────
        df["date"] = df["date"].astype(str).str.strip()
        # Try a few common Indian bank date formats explicitly, then dayfirst=True
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
                    "%Y-%m-%d", "%m/%d/%Y", "%d %b %Y", "%d %B %Y"):
            try:
                parsed = pd.to_datetime(df["date"], format=fmt, errors="coerce")
                # Accept if more than 50 % parsed successfully
                if parsed.notna().mean() > 0.5:
                    df["date"] = parsed
                    break
            except Exception:
                continue
        else:
            # Last resort: let pandas infer
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

        null_dates = df["date"].isna().sum()
        if null_dates:
            warnings.append(f"Dropped {null_dates} row(s) with unparseable dates.")

        # ── 2. Amount coercion ────────────────────────────────────────────
        df["amount"] = (
            df["amount"].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^\d.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

        null_amounts = df["amount"].isna().sum()
        if null_amounts:
            warnings.append(f"Dropped {null_amounts} row(s) with unparseable amounts.")

        # ── 3. Transaction type ───────────────────────────────────────────
        if "transaction_type" not in df.columns:
            df["transaction_type"] = np.where(df["amount"] >= 0, "credit", "debit")

        # ── 4. Raw text ───────────────────────────────────────────────────
        df["raw_text"] = df.get("description", pd.Series("", index=df.index)).astype(str)

        # ── 5. Clean description ──────────────────────────────────────────
        df["description"] = df["raw_text"].str.strip()

        # ── 6. Drop bad rows ──────────────────────────────────────────────
        before = len(df)
        df = df.dropna(subset=["date", "amount"])
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            warnings.append(f"Removed {dropped} duplicate/invalid row(s).")

        # ── 7. Sort & reset index ─────────────────────────────────────────
        df = df.sort_values("date").reset_index(drop=True)

        # ── 8. Final column order ─────────────────────────────────────────
        priority = ["date", "description", "amount", "transaction_type", "raw_text"]
        rest = [c for c in df.columns if c not in priority]
        df = df[priority + rest]

        return df

