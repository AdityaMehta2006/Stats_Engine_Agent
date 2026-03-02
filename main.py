"""
Stat Engine Agent -- Streamlit Frontend.
Upload CSV files and get automated statistical analysis reports.
"""

import os
import sys
import numpy as np
import streamlit as st
from pathlib import Path

# Global seed for reproducible statistical tests and charts
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_settings
from src.utils.security import validate_upload, sanitize_filename


def init_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Stat Engine Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with info and settings. Returns the selected analysis mode."""
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "**Stat Engine Agent** uses AI agents to automatically "
            "analyze your CSV data, run statistical tests, and generate reports."
        )

        st.markdown("---")

        # Analysis mode toggle
        st.markdown("### Analysis Mode")
        regression_mode = st.toggle(
            "🔬 Linear Regression Mode",
            value=False,
            help="When ON, the pipeline focuses on OLS regression analysis: "
                 "R², VIF multicollinearity, ACF/PACF autocorrelation, "
                 "residual diagnostics, and regression-specific charts.",
        )
        if regression_mode:
            st.info(
                "**Regression Mode Active**\n\n"
                "The analysis will focus on:\n"
                "- OLS Linear Regression\n"
                "- VIF (Multicollinearity)\n"
                "- ACF/PACF (Autocorrelation)\n"
                "- Residual Diagnostics\n"
                "- Coefficient Analysis"
            )

        st.markdown("---")
        st.markdown("### Pipeline Steps")
        steps = [
            "1. Schema Analysis",
            "2. Data Cleaning",
            "3. Database Loading",
            "4. Statistical Testing" if not regression_mode else "4. Regression Analysis",
            "5. Report Generation",
        ]
        for step in steps:
            st.markdown(f"- {step}")

        st.markdown("---")
        st.markdown("### Tech Stack")
        st.markdown("- CrewAI + Gemini 2.0 Flash")
        st.markdown("- MySQL (per-CSV databases)")
        st.markdown("- SciPy + Seaborn")
        if regression_mode:
            st.markdown("- Statsmodels (OLS/VIF/ACF)")

    return "regression" if regression_mode else "general"


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to the uploads directory and return the path."""
    settings = get_settings()
    safe_name = sanitize_filename(uploaded_file.name)
    file_path = settings.uploads_dir / safe_name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def display_report(report_content: str):
    """Display the AI-generated report."""
    st.markdown("---")
    st.markdown("## Analysis Report")
    st.markdown(report_content)


def display_charts(csv_path: str, charts_dir: str, analysis_mode: str = "general"):
    """Auto-generate and display interactive Plotly charts."""
    from src.utils.auto_charts import generate_charts, generate_regression_charts, save_charts_as_png

    # Find the cleaned CSV (data_cleaner saves as cleaned_*)
    from pathlib import Path
    csv_dir = Path(csv_path).parent
    cleaned_files = sorted(csv_dir.glob("cleaned_*.csv"))
    data_path = str(cleaned_files[0]) if cleaned_files else csv_path

    with st.spinner("Generating interactive charts..."):
        if analysis_mode == "regression":
            charts = generate_regression_charts(data_path)
        else:
            charts = generate_charts(data_path)

    if not charts:
        st.warning("Not enough data to generate charts.")
        return

    # Save PNGs to the run folder for offline reference
    save_charts_as_png(charts, charts_dir)

    # Display interactive charts in tabs
    st.markdown("---")
    chart_title = "## Regression Charts" if analysis_mode == "regression" else "## Interactive Charts"
    st.markdown(chart_title)

    tab_names = [c["title"] for c in charts]
    tabs = st.tabs(tab_names)
    for tab, chart in zip(tabs, charts):
        with tab:
            st.plotly_chart(chart["fig"], use_container_width=True)
            st.caption(chart["description"])


def generate_pdf_report(report_text: str, charts_dir: str, output_path: str):
    """Generate a PDF report with embedded charts."""
    try:
        from src.utils.pdf_report import generate_pdf
        generate_pdf(report_text, charts_dir, output_path)
        return True
    except ImportError:
        return False
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")
        return False


def main():
    """Main application."""
    init_page()
    analysis_mode = render_sidebar()

    # Header
    st.markdown('<p class="main-header">Stat Engine Agent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a CSV file and let AI agents analyze your data</p>',
        unsafe_allow_html=True,
    )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Maximum file size: 50 MB",
    )

    if uploaded_file is not None:
        # Validate
        settings = get_settings()
        error = validate_upload(
            filename=uploaded_file.name,
            file_size_bytes=uploaded_file.size,
            max_size_mb=settings.max_upload_size_mb,
        )

        if error:
            st.error(f"Upload rejected: {error}")
            return

        # Show file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            mode_label = "🔬 Regression" if analysis_mode == "regression" else "📊 General"
            st.metric("Mode", mode_label)

        # Run analysis button
        if st.button("Run Analysis", type="primary", use_container_width=True):
            # Clean up old uploads to prevent data bleed between runs
            settings = get_settings()
            uploads_dir = settings.uploads_dir
            for old_file in uploads_dir.glob("*.csv"):
                try:
                    old_file.unlink()
                except Exception:
                    pass

            # Save file
            csv_path = save_uploaded_file(uploaded_file)

            # Run the crew
            with st.spinner("Running AI analysis pipeline... This may take a few minutes."):
                progress = st.progress(0)
                status = st.empty()

                status.markdown("**Step 1/5:** Schema Analyst is reading your data...")
                progress.progress(10)

                try:
                    from src.crew import run_analysis
                    result = run_analysis(
                        csv_path,
                        status_callback=status.markdown,
                        analysis_mode=analysis_mode,
                    )
                    progress.progress(90)
                    status.markdown("**Generating charts...**")

                    report_text = result["report"]
                    charts_dir = result["charts_dir"]
                    run_dir = result["run_dir"]

                    # Display the AI report
                    display_report(report_text)

                    # Auto-generate and display interactive charts
                    display_charts(csv_path, charts_dir, analysis_mode=analysis_mode)

                    progress.progress(100)
                    status.markdown("**All done!**")

                    # Download buttons
                    dl_col1, dl_col2 = st.columns(2)

                    with dl_col1:
                        st.download_button(
                            label="📝 Download Report (Markdown)",
                            data=report_text,
                            file_name="stat_engine_report.md",
                            mime="text/markdown",
                        )

                    with dl_col2:
                        # Generate and offer PDF download
                        pdf_path = os.path.join(run_dir, "report.pdf")
                        if generate_pdf_report(report_text, charts_dir, pdf_path):
                            with open(pdf_path, "rb") as pdf_file:
                                st.download_button(
                                    label="📄 Download Report (PDF)",
                                    data=pdf_file.read(),
                                    file_name="stat_engine_report.pdf",
                                    mime="application/pdf",
                                )

                    # Show output location
                    st.success(f"All outputs saved to: {run_dir}")

                    # Clean up uploaded CSVs after successful run
                    for f in uploads_dir.glob("*.csv"):
                        try:
                            f.unlink()
                        except Exception:
                            pass

                except Exception as e:
                    progress.progress(100)
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)

    else:
        # Show placeholder
        st.info("Upload a CSV file to get started. The AI agents will automatically analyze your data.")

        with st.expander("What happens when you upload?"):
            st.markdown("""
            1. **Schema Analyst** reads your CSV, identifies column types, and flags issues
            2. **Data Cleaner** handles missing values, duplicates, and outliers
            3. **DB Architect** designs a relational schema and loads data into MySQL
            4. **Statistical Analyst** runs appropriate tests (t-tests, ANOVA, chi-squared, etc.)
            5. **Report Generator** creates charts and compiles a professional report
            """)

        with st.expander("🔬 What does Regression Mode do?"):
            st.markdown("""
            When toggled ON, step 4 switches to **OLS Linear Regression** analysis:
            - **R² and Adjusted R²** — overall model fit
            - **VIF (Variance Inflation Factor)** — multicollinearity check
            - **ACF / PACF** — residual autocorrelation patterns
            - **Durbin-Watson** — autocorrelation statistic
            - **Jarque-Bera** — residual normality test
            - **Breusch-Pagan** — heteroscedasticity test
            - **Coefficient Analysis** — significance and confidence intervals
            """)


if __name__ == "__main__":
    main()
