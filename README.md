# Spline Basis ↔ Waveform Playground

A tiny app to explore how natural cubic spline coefficients map to output waveforms, compare two hand-drawn waveforms via least-squares fits, and visualise uncertainty by sampling coefficients.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
