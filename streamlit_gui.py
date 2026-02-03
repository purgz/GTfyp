import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chart_studio import plotly
import plotly.tools as tls
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import pandas as pd
import streamlit.components.v1 as components

# Now you can import your modules
from source import simulation
from source.simulation import Games
from source.simulation import quaternary_plot
from source.simulation import plotting
# -----------------
plt.style.use(['science','no-latex'])
st.set_page_config(page_title="Evolutionary Game Dynamics Lab", layout="wide")


def simplex_vertices():
    barycentric_coords = np.array([
        [1, 0, 0, 0],  # R
        [0, 1, 0, 0],  # P
        [0, 0, 1, 0],  # S
        [0, 0, 0, 1],  # L
    ])

    # Cartersian coords for corners of pyramid
    return plotting.get_cartesian_array_from_barycentric(barycentric_coords)


def add_simplex_edges(fig, line_width=3, line_opacity=0.9, line_color="black"):

    verts = simplex_vertices()

    edges = [(0, 1), (0, 2), (0, 3),
             (1, 2), (1, 3),
             (2, 3)]
    
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[verts[i][0], verts[j][0]],
            y=[verts[i][1], verts[j][1]],
            z=[verts[i][2], verts[j][2]],
            mode="lines",
            line={"color": line_color, "width": line_width},
            opacity=line_opacity,
            hoverinfo="skip",
            showlegend=False
        ))

def add_corner_labels(fig, font_size=12):
    verts = simplex_vertices()
    labels = ["R", "P", "S", "L"]

    fig.add_trace(go.Scatter3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        mode="text",
        text=labels,
        textfont={"size": font_size, "color": "black"},
        hoverinfo="skip",
        showlegend=False
    ))

def style_simplex_scene(fig):
    
    fig.update_layout(
        height=720,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene=dict(
            bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.35, z=1.15)  # tweak if you want “closer”
            ),
        ),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.75)"
        )
    )



# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("Simulation Parameters")
pop_size = st.sidebar.number_input("Population Size (N)", value=100, step=10)
iterations = st.sidebar.number_input("Iterations per Simulation", value=10000, step=1000)
num_sims = st.sidebar.number_input("Number of Simulations", value=100, step=10)
w = st.sidebar.slider("Selection Strength (w)", 0.0, 1.0, 0.45)
process = st.sidebar.selectbox("Interaction Process", ["Moran", "Local"])

# --- MAIN: MATRIX INPUT ---

default_matrix = np.array([
    [0.0, -0.8, 1.0, 0.2],
    [1.0, 0.0, -0.8, 0.2],
    [-0.8, 1.0, 0.0, 0.2],
    [0.2, 0.2, 0.2, 0.0]
])

# Use data_editor for an interactive matrix UI
matrix_df = pd.DataFrame(default_matrix, 
                         columns=["R/C", "P/D", "S/L", "Q"], 
                         index=["R/C", "P/D", "S/L", "Q"])
#edited_df = st.data_editor(matrix_df)


# Compact "matrix-like" editor
matrix_col, metrics_col = st.columns([1, 2], vertical_alignment="top")

with matrix_col:
    st.subheader("Payoff Matrix")
    edited_df = st.data_editor(
        matrix_df,
        use_container_width=False,
        width=380,     # tweak to taste
        height=210,    # keeps it looking like a 4x4 matrix
        num_rows="fixed",
    )

matrix = edited_df.to_numpy()


# --- EXECUTION ---
if st.button("▶️ Run Simulation"):
    with st.spinner("Running Moran Process..."):
        # Calling the function from your sim.py
        if process == "Moran":
            
          delta_h, delta_rps, avg_traj, all_traj = simulation.moran_batch_sim(
              pop_size=pop_size,
              iterations=iterations,
              simulations=num_sims,
              w=w,
              matrix=matrix,
              point_cloud=False,
              initial_rand=False,
              traj=True
          )
        elif process == "Local":
            delta_h, delta_rps, avg_traj, all_traj = simulation.local_batch_sim(
              pop_size=pop_size,
              iterations=iterations,
              simulations=num_sims,
              w=w,
              matrix=matrix,
              point_cloud=False,
              initial_rand=False,
              traj=True
          )
        elif process == "Fermi":
          delta_h, delta_rps, avg_traj, all_traj = simulation.fermi_batch_sim(
              pop_size=pop_size,
              iterations=iterations,
              simulations=num_sims,
              w=w,
              matrix=matrix,
              point_cloud=False,
              initial_rand=False,
              traj=True
          )

    with metrics_col:
        st.subheader("Summary")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Mean ΔH (Snowdrift)", f"{delta_h:.8f}")
        with m2:
            st.metric("Mean ΔH (RPS)", f"{delta_rps:.8f}")

    # --- PLOTTING ---
    st.divider()
    st.subheader("Visualizations")
    
    plot_type = st.radio("Select Plot", ["Trajectory (3D/Quaternary)", "Time Series"])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if plot_type == "Trajectory (3D/Quaternary)":
        df = pd.DataFrame({"R": avg_traj[0], "P": avg_traj[1], "S": avg_traj[2], "L": avg_traj[3]})
        #st.write(df.head())
        #fig = quaternary_plot([df], labels=[process], show=False)
        
        carts = plotting.get_cartesian_array_from_barycentric(df.values)

        fig = go.Figure()
        add_corner_labels(fig)
        add_simplex_edges(fig)
        
        
        fig.add_trace(go.Scatter3d(
            x=carts[:,0],
            y=carts[:,1],
            z=carts[:,2],
            mode="lines",
            line={"color": "blue", "width": 5},
            name=f"{process} Trajectory",
        ))

        style_simplex_scene(fig)
        st.subheader(f"{process} Trajectory")
        st.plotly_chart(fig, use_container_width=True)


    else:
        pass

if st.button("Point cloud animation"):
    with st.spinner("Running Process..."):
        # Calling the function from your sim.py
        if process == "Moran":
            
          delta_h, delta_rps, avg_traj, all_traj = simulation.moran_batch_sim(
              pop_size=pop_size,
              iterations=iterations,
              simulations=num_sims,
              w=w,
              matrix=matrix,
              point_cloud=True,
              initial_rand=True,
              traj=False
          )
        elif process == "Local":
            delta_h, delta_rps, avg_traj, all_traj = simulation.local_batch_sim(
              pop_size=pop_size,
              iterations=iterations,
              simulations=num_sims,
              w=w,
              matrix=matrix,
              point_cloud=False,
              initial_rand=False,
              traj=True
          )
        elif process == "Fermi":
          delta_h, delta_rps, avg_traj, all_traj = simulation.fermi_batch_sim(
              pop_size=pop_size,
              iterations=iterations,
              simulations=num_sims,
              w=w,
              matrix=matrix,
              point_cloud=False,
              initial_rand=False,
              traj=True
          )

        
        st.session_state["all_traj"] = all_traj
        st.session_state["process"] = process
        st.session_state["iterations"] = iterations



import plotly.graph_objects as go

def pointcloud_animation_figure(all_traj):
    num_frames = all_traj.shape[2]

    # initial frame
    carts0 = plotting.get_cartesian_array_from_barycentric(all_traj[:, :, 0])

    fig = go.Figure()

    add_corner_labels(fig)
    add_simplex_edges(fig)

    # Initial scatter points
    fig.add_trace(go.Scatter3d(
        x=carts0[:, 0], y=carts0[:, 1], z=carts0[:, 2],
        mode="markers",
        marker=dict(size=4, opacity=0.9, color="blue"),
        name="Simulations",
        showlegend=True
    ))

    # Get cartesian coors for each frame 
    scatter_index = len(fig.data) - 1

    frames = []
    for k in range(num_frames):
        carts = plotting.get_cartesian_array_from_barycentric(all_traj[:, :, k])
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=carts[:, 0], y=carts[:, 1], z=carts[:, 2],
                mode="markers",
                marker=dict(size=4, opacity=0.9, color="blue")
            )],
            traces=[scatter_index],
            name=str(k)
        ))

    fig.frames = frames

    # slider + play button
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.02, y=1.08,
            bgcolor="rgba(255,255,255,1)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12, color="black"),
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 40, "redraw": True},
                                  "transition": {"duration": 0},
                                  "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            bgcolor="rgba(255,255,255,1)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12, color="black"),
            x=0.1, y=0.02, len=0.85,
            currentvalue={"prefix": "Frame: "},
            steps=[dict(method="animate",
                        args=[[str(k)], {"mode": "immediate",
                                        "frame": {"duration": 0, "redraw": True},
                                        "transition": {"duration": 0}}],
                        label=str(k))
                   for k in range(num_frames)]
        )],
    )

    style_simplex_scene(fig)
    return fig



all_traj = st.session_state.get("all_traj", None)

if all_traj is not None:
    
    fig = pointcloud_animation_figure(all_traj)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# --- COMPARISON WITH ANALYTICAL (Optional) ---
with st.expander("Compare with Analytical Integration (nquad)"):
    st.write("Click to compute the theoretical expected ΔH using the nquad method.")
    if st.button("Calculate Theoretical"):
        # Here you would call your nquad logic from aug_rps.py
        st.info("Integrating over the simplex... this may take a moment.")
        # Placeholder for your nquad call:
        # res, err = nquad(...) 
        # st.write(f"Theoretical result: {res}")
        st.warning("Integration logic needs to be linked from aug_rps.py")