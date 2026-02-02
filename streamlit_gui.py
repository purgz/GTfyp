import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


# Now you can import your modules
from source import simulation
from source.simulation import Games
from source.simulation import quaternary_plot
# -----------------

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
        mean_h, mean_rps, avg_traj, all_traj = simulation.run_simulation(
            pop_size=pop_size,
            iterations=iterations,
            simulations=num_sims,
            w=w,
            matrix=matrix,
            process=process,
            point_cloud=True
        )

    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean ΔH (Snowdrift)", f"{mean_h:.8f}")
        st.write("This is the average change in your Hamiltonian over the total simulations.")
        
    with col2:
        st.metric("Mean ΔH (RPS)", f"{mean_rps:.8f}")

    # --- PLOTTING ---
    st.divider()
    st.subheader("Visualizations")
    
    plot_type = st.radio("Select Plot", ["Trajectory (3D/Quaternary)", "Time Series"])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if plot_type == "Trajectory (3D/Quaternary)":
        # Using your custom quaternary_plot from plotting.py
        # You might need to adjust the input format to match your plotting.py function
        # For now, showing a simple 3D trajectory plot as a placeholder
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(avg_traj[0], avg_traj[1], avg_traj[2])
        ax.set_xlabel("Strategy 1")
        ax.set_ylabel("Strategy 2")
        ax.set_zlabel("Strategy 3")
        st.pyplot(fig)
        
    else:
        # Time series of frequencies
        for i in range(avg_traj.shape[0]):
            ax.plot(avg_traj[i], label=f"Strategy {i+1}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

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