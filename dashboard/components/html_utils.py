"""
HTML Rendering Utility for Hyderabad PM2.5 Dashboard
Renders raw HTML cleanly without triggering Markdown code block formatting.
"""

from typing import Optional, Any
import streamlit as st


def render_html(html: str, container: Optional[Any] = None) -> None:
    """
    Renders raw HTML cleanly without triggering Markdown code block formatting.
    Strips each line, drops empty lines and lines starting with '<!--',
    joins the rest with a single space, and renders the cleaned HTML.
    """
    target = container or st
    lines = [line.strip() for line in html.splitlines()]
    cleaned_lines = [
        line for line in lines
        if line and not line.startswith("<!--")
    ]
    cleaned = " ".join(cleaned_lines)
    if hasattr(target, "html"):
        target.html(cleaned)
    else:
        target.markdown(cleaned, unsafe_allow_html=True)
