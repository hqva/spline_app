# app.py
from __future__ import annotations
import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from spline import (
    EMULATOR_N_TIMESTEPS,
    EMULATOR_PRESSURE_SPLINE_BASE,
    generate_spline_dmatrix,
    fit_coefficients,
    reconstruct_waveform,
    sample_waveforms,
    generate_interior_knots_logspaced,
)


st.set_page_config(page_title="Spline Basis ↔ Waveform", layout="wide")


# ---- helpers ----
def time_axis(n=EMULATOR_N_TIMESTEPS) -> np.ndarray:
    return np.linspace(0, 1, num=n, endpoint=False)


def _collapse_strokes_to_series(
    json_data, width: int, height: int, n=EMULATOR_N_TIMESTEPS
):
    """Use stroke vectors not raster: robust with drawable-canvas."""
    if not json_data or "objects" not in json_data:
        return None
    xs = [[] for _ in range(width)]
    for obj in json_data["objects"]:
        if obj.get("type") != "path":
            continue
        # obj['path'] is a list of path commands; use the cachedPoints (if present) for speed
        pts = obj.get("path", [])
        # Fallback: fabric.js stores points relative; use 'path' decoding
        curr_x, curr_y = 0.0, 0.0
        for seg in pts:
            if not seg:
                continue
            cmd = seg[0]
            if cmd == "M" or cmd == "L":
                # absolute move/line: seg = ["M", x, y] or ["L", x, y]
                px = seg[1]
                py = seg[2]
                curr_x, curr_y = px, py
                x_pix = int(np.clip(round(px), 0, width - 1))
                xs[x_pix].append(py)
            elif cmd == "Q" and len(seg) == 5:
                # quadratic curve: sample end point
                px = seg[3]
                py = seg[4]
                curr_x, curr_y = px, py
                x_pix = int(np.clip(round(px), 0, width - 1))
                xs[x_pix].append(py)
            # other commands ignored for simplicity

    # average y per x; carry fills
    col = np.full(width, np.nan, dtype=float)
    for i in range(width):
        if xs[i]:
            col[i] = float(np.mean(xs[i]))

    # forward/back fill
    # forward:
    last = np.nan
    for i in range(width):
        if np.isnan(col[i]) and not np.isnan(last):
            col[i] = last
        elif not np.isnan(col[i]):
            last = col[i]
    # backward:
    last = np.nan
    for i in range(width - 1, -1, -1):
        if np.isnan(col[i]) and not np.isnan(last):
            col[i] = last
        elif not np.isnan(col[i]):
            last = col[i]

    if np.isnan(col).all():
        return None

    # normalise 0..1 and invert y (top=1)
    ymin, ymax = np.nanmin(col), np.nanmax(col)
    if ymax - ymin < 1e-9:
        col[:] = 0.5
    else:
        col = (col - ymin) / (ymax - ymin)
    col = 1.0 - col

    # resample to n points
    t_src = np.linspace(0, 1, num=width, endpoint=False)
    t_dst = np.linspace(0, 1, num=n, endpoint=False)
    return np.interp(t_dst, t_src, col)


def vector_or_raster_to_waveform(canvas_result, n=EMULATOR_N_TIMESTEPS):
    # try vector first
    y = None
    jd = canvas_result.json_data
    if jd:
        try:
            y = _collapse_strokes_to_series(
                jd,
                width=int(canvas_result.image_data.shape[1]),
                height=int(canvas_result.image_data.shape[0]),
                n=n,
            )
        except Exception:
            y = None
    # fallback to raster (alpha channel)
    if y is None and canvas_result.image_data is not None:
        img = canvas_result.image_data  # H, W, 4
        h, w, _ = img.shape
        alpha = img[..., 3]
        ys = np.zeros(w, dtype=float)
        has = np.zeros(w, dtype=bool)
        for x in range(w):
            ycoords = np.where(alpha[:, x] > 0)[0]
            if ycoords.size > 0:
                ys[x] = ycoords.mean()
                has[x] = True
        # forward/back fill
        last = None
        for x in range(w):
            if has[x]:
                last = ys[x]
            elif last is not None:
                ys[x] = last
                has[x] = True
        last = None
        for x in range(w - 1, -1, -1):
            if has[x]:
                last = ys[x]
            elif last is not None:
                ys[x] = last
                has[x] = True
        if has.any():
            ys = (ys - ys.min()) / max(1e-9, ys.max() - ys.min())
            ys = 1.0 - ys
            t_src = np.linspace(0, 1, num=w, endpoint=False)
            t_dst = np.linspace(0, 1, num=n, endpoint=False)
            y = np.interp(t_dst, t_src, ys)
    return y


def plot_waveforms(
    time, curves, names=None, mode="lines", opacity=1.0, height=420
):
    fig = go.Figure()
    for i, y in enumerate(curves):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=y,
                mode=mode,
                name=None if names is None else names[i],
                opacity=opacity
                if isinstance(opacity, (int, float))
                else opacity[i],
            )
        )
    fig.update_layout(
        height=height,  # <- was 420 fixed
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="time (normalised)",
        yaxis_title="amplitude",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
        ),
    )
    return fig


def plot_coefficients(mu, sigma=None):
    idx = np.arange(mu.size)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx, y=mu, name="μ"))
    if sigma is not None:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([idx, idx[::-1]]),
                y=np.concatenate([mu + sigma, (mu - sigma)[::-1]]),
                mode="lines",
                name="±σ",
                fill="toself",
                opacity=0.25,
            )
        )
    fig.update_layout(
        barmode="overlay",
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="basis index",
        yaxis_title="value",
    )
    return fig


# ---- sidebar controls ----
st.sidebar.header("Spline")
n_knots = st.sidebar.number_input(
    "n_knots (incl. ends)", min_value=6, max_value=30, value=16, step=1
)
base = st.sidebar.number_input(
    "log-spacing base",
    min_value=1.1,
    max_value=10.0,
    value=float(EMULATOR_PRESSURE_SPLINE_BASE),
    step=0.1,
)
include_intercept = st.sidebar.selectbox(
    "Include intercept?", options=[True, False], index=0
)

st.sidebar.header("Sampling")
n_samples = st.sidebar.number_input(
    "Samples", min_value=0, max_value=2000, value=10, step=10
)
seed = st.sidebar.number_input(
    "Random seed", min_value=0, max_value=10_000_000, value=42, step=1
)

# ---- build design matrix ----
X = generate_spline_dmatrix(
    n_knots=n_knots, base=base, include_intercept=include_intercept
)
n_coeff = X.shape[1]
t = time_axis()

# positions of knots along the time axis (include start/end for context)
_interior = generate_interior_knots_logspaced(
    n_knots=n_knots, base=base, time=t
)
knot_pos = np.concatenate(([t[0]], _interior, [t[-1]]))  # shape (n_knots,)

# ---- session init ----
if "mu" not in st.session_state or st.session_state.mu.size != n_coeff:
    st.session_state.mu = np.zeros(n_coeff)
if "sigma" not in st.session_state or st.session_state.sigma.size != n_coeff:
    st.session_state.sigma = np.zeros(n_coeff)

# ---- layout: left (sliders), right (draw & outputs) ----
left, right = st.columns([1.1, 1.9])

with left:
    st.subheader("Coefficients")
    # quick σ all
    sigma_all = st.slider("Set all σ", 0.0, 2.0, 0.0, 0.01)
    if sigma_all is not None:
        st.session_state.sigma[:] = sigma_all

    # sliders grid
    for i in range(n_coeff):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.mu[i] = st.slider(
                f"μ[{i}]", -5.0, 5.0, float(st.session_state.mu[i]), 0.01
            )
        with c2:
            st.session_state.sigma[i] = st.slider(
                f"σ[{i}]", 0.0, 2.0, float(st.session_state.sigma[i]), 0.01
            )

    st.plotly_chart(
        plot_coefficients(st.session_state.mu, st.session_state.sigma),
        use_container_width=True,
    )

with right:
    st.subheader("Draw waveform → Fit coefficients")

    # two side-by-side areas: drawing (left) and tiny preview (right)
    draw_col, prev_col = st.columns([1.6, 1.0])

    with draw_col:
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=5,
            stroke_color="#1f77b4",
            background_color="#ffffff",
            update_streamlit=True,
            display_toolbar=True,
            height=220,
            width=700,
            drawing_mode="freedraw",
            key="canvas_one",
        )

    with prev_col:
        y_draw = vector_or_raster_to_waveform(canvas, n=EMULATOR_N_TIMESTEPS)
        if y_draw is not None:
            st.markdown("Preview")
            st.plotly_chart(
                plot_waveforms(t, [y_draw], names=["Drawn"], height=160),
                use_container_width=True,
            )
        else:
            y_draw = None

    # ⬇️ buttons placed at the SAME nesting level as draw_col/prev_col
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        fit_btn = st.button("Fit μ from drawing")
    with btn_col2:
        clear_btn = st.button("Clear drawing")

    if clear_btn:
        st.rerun()

    if fit_btn and y_draw is not None:
        st.session_state.mu = fit_coefficients(X, y_draw)

    # outputs
    mu = st.session_state.mu
    sigma = st.session_state.sigma
    y_mu = reconstruct_waveform(X, mu)

    curves = [y_mu]
    names = ["mean(μ)"]
    opac = [1.0]

    if n_samples > 0 and sigma.max() > 0:
        rng = np.random.default_rng(int(seed))
        samples = sample_waveforms(X, mu, sigma, n_samples=n_samples, rng=rng)
        overlay_n = min(200, n_samples)
        for i in range(overlay_n):
            curves.append(samples[i])
            names.append(f"s{i}")
            opac.append(0.12)

    fig = plot_waveforms(t, curves, names=names, opacity=opac)

    # evaluate the mean waveform at the knot positions
    y_mu_at_knots = np.interp(knot_pos, t, y_mu)

    # overlay knot markers (open circles so they’re visible over lines)
    fig.add_trace(
        go.Scatter(
            x=knot_pos,
            y=y_mu_at_knots,
            mode="markers",
            name="spline knots",
            marker=dict(size=8, symbol="circle-open"),
            hovertemplate="t=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )

    st.plotly_chart(fig, use_container_width=True)
