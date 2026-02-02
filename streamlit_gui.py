import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chart_studio import plotly
import plotly.tools as tls
import plotly.express as px
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


# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("Simulation Parameters")
pop_size = st.sidebar.number_input("Population Size (N)", value=100, step=10)
iterations = st.sidebar.number_input("Iterations per Simulation", value=10000, step=1000)
num_sims = st.sidebar.number_input("Number of Simulations", value=100, step=10)
w = st.sidebar.slider("Selection Strength (w)", 0.0, 1.0, 0.45)
process = st.sidebar.selectbox("Interaction Process", ["Moran", "Local"])

# --- MAIN: MATRIX INPUT ---
st.subheader("Payoff Matrix")
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
edited_df = st.data_editor(matrix_df)
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

    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean ΔH (Snowdrift)", f"{delta_h:.8f}")
        
    with col2:
        st.metric("Mean ΔH (RPS)", f"{delta_rps:.8f}")

    # --- PLOTTING ---
    st.divider()
    st.subheader("Visualizations")
    
    plot_type = st.radio("Select Plot", ["Trajectory (3D/Quaternary)", "Time Series"])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if plot_type == "Trajectory (3D/Quaternary)":
        df = pd.DataFrame({"R": avg_traj[0], "P": avg_traj[1], "S": avg_traj[2], "L": avg_traj[3]})
        st.write(df.head())
        #fig = quaternary_plot([df], labels=[process], show=False)
        
        carts = plotting.get_cartesian_array_from_barycentric(df.values)
        print(carts)
        plotly_fig = px.line_3d(carts, x=0, y =1, z=2)
        
        st.plotly_chart(plotly_fig)


    else:
        pass

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