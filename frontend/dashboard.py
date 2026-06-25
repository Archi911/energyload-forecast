# import streamlit as st
# import requests
# import pandas as pd
# import plotly.graph_objects as go
# import os

# # Page Setup
# st.set_page_config(page_title="VoltHive Grid Console", layout="wide")

# st.markdown("""
#     <style>
#     .main { background-color: #060913 !important; }
#     h1, h2, h3, p { color: #f8fafc !important; }
#     div[data-testid="stMetricValue"] { color: #22d3ee !important; font-family: monospace; font-size: 28px; }
#     /* Making standard buttons look like nice grid tiles */
#     .stButton > button { width: 100%; height: 50px; font-family: monospace; font-size: 13px; font-weight: bold; }
#     </style>
# """, unsafe_allow_html=True)

# mapping_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "household_cluster_mapping.csv")
# registry_df = pd.read_csv(mapping_file_path)

# st.title("VoltHive: Master Substation Console")
# st.caption("Live AI MoE Inference Stream | Decentralized Node Controller")
# st.markdown("---")

# col_nav1, col_nav2 = st.columns([1, 2])
# with col_nav1:
#     cluster_selection = st.selectbox("Select Substation Cluster:", [0, 1, 2], index=1)
#     filtered_roster = registry_df[registry_df['cluster_id'] == cluster_selection]['LCLid'].unique().tolist()

# if "selected_house" not in st.session_state or st.session_state.selected_house not in filtered_roster:
#     st.session_state.selected_house = filtered_roster[0]

# with col_nav2:
#     selected_dropdown_house = st.selectbox("Search & Isolate Household Node:", filtered_roster, index=filtered_roster.index(st.session_state.selected_house))
#     if selected_dropdown_house != st.session_state.selected_house:
#         st.session_state.selected_house = selected_dropdown_house
#         st.rerun()

# try:
#     with st.spinner("Fetching Live Tensor Streams from Supabase..."):
#         house_res = requests.get(f"http://127.0.0.1:8000/predict/household/{st.session_state.selected_house}")
#         cluster_res = requests.get(f"http://127.0.0.1:8000/predict/cluster/{cluster_selection}")
    
#     if house_res.status_code == 200 and cluster_res.status_code == 200:
#         h_data = house_res.json()
#         c_data = cluster_res.json()
        
#         m1, m2, m3, m4 = st.columns(4)
#         m1.metric("Target Node", h_data["lcl_id"])
#         m2.metric("Current Load", f"{h_data['current_load']:.3f} kWh")
#         m3.metric("Peak Forecast", f"{h_data['peak_load']:.3f} kWh")
#         m4.metric("Total Swarm Capacity", f"{c_data['aggregated_tomorrow_kwh']:,} kWh")
        
#         col_grid, col_visuals = st.columns([5, 4])
        
#         with col_grid:
#             st.subheader(f"Active Grid Matrix ({c_data['total_nodes']} Total Nodes)")
            
#             # Pagination Engine
#             nodes_per_page = 48
#             total_pages = (len(filtered_roster) // nodes_per_page) + 1
            
#             page_col, blank_col = st.columns([1, 3])
#             with page_col:
#                 current_page = st.number_input("Grid Page", min_value=1, max_value=total_pages, value=1)
            
#             start_idx = (current_page - 1) * nodes_per_page
#             display_slice = filtered_roster[start_idx : start_idx + nodes_per_page]
            
#             st.caption(f"Displaying nodes {start_idx + 1} to {min(start_idx + nodes_per_page, len(filtered_roster))}")
            
#             # Native Streamlit Buttons
#             for r_idx in range(0, len(display_slice), 8):
#                 chunk = display_slice[r_idx:r_idx+8]
#                 cols = st.columns(8)
#                 for c_idx, node_id in enumerate(chunk):
#                     with cols[c_idx]:
#                         # Use Streamlit's native 'primary' type to highlight the active node
#                         btn_type = "primary" if node_id == st.session_state.selected_house else "secondary"
#                         # Strip "MAC00" to make the ID fit perfectly in the square
#                         display_name = node_id.replace("MAC00", "") 
                        
#                         if st.button(display_name, key=f"btn_{node_id}", type=btn_type):
#                             st.session_state.selected_house = node_id
#                             st.rerun()
                            
#         with col_visuals:
#             times = pd.date_range("00:00", periods=48, freq="30min").strftime("%H:%M")
#             tab1, tab2 = st.tabs(["Single Node Forecast", "Substation Capacity"])
            
#             with tab1:
#                 fig_h = go.Figure(go.Scatter(x=times, y=h_data["forecast_48_steps"], mode='lines', line=dict(color='#00ff88', width=3.5)))
#                 fig_h.add_hline(y=1.5, line_dash="dash", line_color="#ef4444", annotation_text="Danger Threshold")
#                 fig_h.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10))
#                 st.plotly_chart(fig_h, use_container_width=True)
                
#             with tab2:
#                 fig_c = go.Figure(go.Scatter(x=times, y=c_data["aggregated_curve"], mode='lines', line=dict(color='#a855f7', width=3.5, shape='spline')))
#                 fig_c.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10))
#                 st.plotly_chart(fig_c, use_container_width=True)
# except Exception as e:
#     st.error(f"Server Communication Error: {e}")

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os
import hashlib

st.set_page_config(page_title="VoltHive Grid Console", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    h1, h2, h3, h4, h5, h6, p, span, label { color: #0f172a !important; }
    
    div[data-testid="stMetricValue"] { color: #0284c7 !important; font-family: monospace; font-size: 26px; font-weight: bold; }
    div[data-testid="stMetricLabel"] p { color: #64748b !important; font-weight: 600; text-transform: uppercase; font-size: 12px; }
    
    .stButton > button { 
        width: 100%; 
        height: 60px; 
        font-family: monospace; 
        font-size: 14px; 
        font-weight: bold;
        border: 1px solid #cbd5e1;
        background-color: #f8fafc;
        color: #0f172a;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover { border-color: #0ea5e9; background-color: #f0f9ff; }
    .stButton > button[data-baseweb="button"]:focus { border-color: #0ea5e9; background-color: #e0f2fe; }
    
    hr { border-color: #e2e8f0; }
    </style>
""", unsafe_allow_html=True)

mapping_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "household_cluster_mapping.csv")
registry_df = pd.read_csv(mapping_file_path)

st.title("VoltHive: Master Substation Console")
st.caption("Live AI MoE Inference Stream | Decentralized Node Controller")
st.markdown("---")

col_nav1, col_nav2 = st.columns([1, 2])
with col_nav1:
    cluster_selection = st.selectbox("Select Substation Cluster:", [0, 1, 2], index=1)
    filtered_roster = registry_df[registry_df['cluster_id'] == cluster_selection]['LCLid'].unique().tolist()

if "selected_house" not in st.session_state or st.session_state.selected_house not in filtered_roster:
    st.session_state.selected_house = filtered_roster[0]

with col_nav2:
    selected_dropdown_house = st.selectbox("Search & Isolate Household Node:", filtered_roster, index=filtered_roster.index(st.session_state.selected_house))
    if selected_dropdown_house != st.session_state.selected_house:
        st.session_state.selected_house = selected_dropdown_house
        st.rerun()

try:
    with st.spinner("Fetching Live Tensor Streams from Supabase..."):
        house_res = requests.get(f"http://127.0.0.1:8000/predict/household/{st.session_state.selected_house}")
        cluster_res = requests.get(f"http://127.0.0.1:8000/predict/cluster/{cluster_selection}")
    
    if house_res.status_code == 200 and cluster_res.status_code == 200:
        h_data = house_res.json()
        c_data = cluster_res.json()
        
        total_predicted_24h = sum(h_data["forecast_48_steps"])
        
        # STRUCTURE FIX: 5 Columns to hold all metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Target Node", h_data["lcl_id"])
        m2.metric("Current Load", f"{h_data['current_load']:.3f} kWh")
        m3.metric("Peak Forecast", f"{h_data['peak_load']:.3f} kWh")
        m4.metric("Node 24h Total", f"{total_predicted_24h:.2f} kWh")
        m5.metric("Swarm 24h Capacity", f"{c_data['aggregated_tomorrow_kwh']:,} kWh")
        
        st.markdown("---")
        
        col_grid, col_visuals = st.columns([5, 4])
        
        with col_grid:
            st.subheader(f"Active Grid Matrix ({c_data['total_nodes']} Total Nodes)")
            
            nodes_per_page = 48
            total_pages = (len(filtered_roster) // nodes_per_page) + 1
            
            page_col, blank_col = st.columns([1, 3])
            with page_col:
                current_page = st.number_input("Grid Page", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (current_page - 1) * nodes_per_page
            display_slice = filtered_roster[start_idx : start_idx + nodes_per_page]
            
            st.caption(f"Displaying nodes {start_idx + 1} to {min(start_idx + nodes_per_page, len(filtered_roster))}. Heatmap indicators show simulated baseline intensity.")
            
            for r_idx in range(0, len(display_slice), 8):
                chunk = display_slice[r_idx:r_idx+8]
                cols = st.columns(8)
                for c_idx, node_id in enumerate(chunk):
                    with cols[c_idx]:
                        load_intensity = int(hashlib.md5(node_id.encode()).hexdigest(), 16) % 100
                        
                        if load_intensity < 33:
                            indicator = "[L]" 
                        elif load_intensity < 66:
                            indicator = "[M]" 
                        else:
                            indicator = "[H]" 
                            
                        display_name = f"{indicator} {node_id.replace('MAC00', '')}"
                        btn_type = "primary" if node_id == st.session_state.selected_house else "secondary"
                        
                        if st.button(display_name, key=f"btn_{node_id}", type=btn_type):
                            st.session_state.selected_house = node_id
                            st.rerun()
                            
        with col_visuals:
            times = pd.date_range("00:00", periods=48, freq="30min").strftime("%H:%M")
            tab1, tab2 = st.tabs(["Single Node Forecast", "Substation Capacity"])
            
            with tab1:
                fig_h = go.Figure(go.Scatter(x=times, y=h_data["forecast_48_steps"], mode='lines', fill='tozeroy', line=dict(color='#0ea5e9', width=3)))
                fig_h.add_hline(y=1.5, line_dash="dash", line_color="#ef4444", annotation_text="Danger Threshold", annotation_font_color="#ef4444")
                
                # GRAPH FIX: Force Y-axis to start at 0 and scale properly past the 1.5 line
                y_max = max(1.6, h_data['peak_load'] * 1.2)
                fig_h.update_yaxes(range=[0, y_max])
                
                fig_h.update_layout(template="plotly_white", height=400, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_h, use_container_width=True)
                
            with tab2:
                fig_c = go.Figure(go.Scatter(x=times, y=c_data["aggregated_curve"], mode='lines', fill='tozeroy', line=dict(color='#8b5cf6', width=3, shape='spline')))
                
                # GRAPH FIX: Force aggregate Y-axis to start at 0
                fig_c.update_yaxes(rangemode="tozero")
                
                fig_c.update_layout(template="plotly_white", height=400, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_c, use_container_width=True)
except Exception as e:
    st.error(f"Server Communication Error: {e}")