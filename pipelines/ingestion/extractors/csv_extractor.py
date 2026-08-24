from __future__ import annotations

import io
import logging
from typing import Any, List, Union

import pandas as pd

from .base import BaseExtractor, ExtractedPage

logger = logging.getLogger("neuroflow-csv-extractor")


class CSVExtractor(BaseExtractor):
    """
    CSV Data Extractor:
    - Uses pandas to read CSV
    - Detects numeric vs text columns
    - For small CSVs (<1000 rows): converts each 100-row block to a markdown table
    - For large CSVs (>=1000 rows): generates statistical summaries (dtypes, min/max/mean, top-5 categorical) + sample rows
    - Each 100-row block becomes one ExtractedPage
    """

    async def extract(self, source: Union[str, bytes, io.BytesIO], **kwargs) -> List[ExtractedPage]:
        if isinstance(source, bytes):
            csv_file = io.BytesIO(source)
        elif isinstance(source, io.BytesIO):
            csv_file = source
        else:
            csv_file = source

        df = pd.read_csv(csv_file)
        results: List[ExtractedPage] = []
        total_rows = len(df)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        text_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        common_meta = {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "numeric_columns": numeric_cols,
            "text_columns": text_cols,
        }

        # For large CSVs (>= 1000 rows), create an overarching statistical summary page
        if total_rows >= 1000:
            summary_sections = [
                f"# CSV Dataset Summary (Total Rows: {total_rows}, Columns: {len(df.columns)})",
                f"**Numeric Columns ({len(numeric_cols)}):** {', '.join(numeric_cols) if numeric_cols else 'None'}",
                f"**Text Columns ({len(text_cols)}):** {', '.join(text_cols) if text_cols else 'None'}",
            ]

            if numeric_cols:
                num_desc = df[numeric_cols].describe().T[["min", "mean", "max"]]
                summary_sections.append("\n### Numeric Summary:\n" + num_desc.to_markdown())

            if text_cols:
                top_cats = []
                for col in text_cols[:5]:  # limit to first 5 text columns
                    top_vals = df[col].value_counts().head(5).to_dict()
                    top_cats.append(f"- **{col}** top values: {top_vals}")
                summary_sections.append("\n### Categorical Top-5 Values:\n" + "\n".join(top_cats))

            sample_head = df.head(5).to_markdown(index=False)
            summary_sections.append(f"\n### Sample First 5 Rows:\n{sample_head}")

            summary_text = "\n\n".join(summary_sections)
            results.append(
                ExtractedPage(
                    page_number=1,
                    content=summary_text,
                    content_type="text",
                    metadata={**common_meta, "is_statistical_summary": True},
                )
            )

        # Slice every 100 rows into an ExtractedPage table
        page_offset = 2 if total_rows >= 1000 else 1
        for i in range(0, total_rows, 100):
            sub_df = df.iloc[i : i + 100]
            try:
                table_md = sub_df.to_markdown(index=False)
            except Exception:
                table_md = sub_df.to_string(index=False)

            page_num = (i // 100) + page_offset
            results.append(
                ExtractedPage(
                    page_number=page_num,
                    content=table_md,
                    content_type="table",
                    metadata={
                        **common_meta,
                        "row_start": i,
                        "row_end": min(i + 100, total_rows),
                    },
                )
            )

        return results
