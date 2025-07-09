#!/usr/bin/env python
# coding: utf-8

# In[110]:


#############################
# Trip-Segment Clustering App
# Fuzzy C-Means Clustering for Transportation Route Analysis
# --------------------------------------
# • 35-cluster Fuzzy C-Means model
# • FPC = 0.9227, Silhouette = 0.276
# • Used downstream in SQL procedures to
#   predict Load-Drive-Unload times
#############################

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import numpy as np

# --------------------------------------------------
# 1. App / page setup
# --------------------------------------------------
st.set_page_config(
    page_title="FCM Trip-Segment Clustering",
    page_icon="🚛",
    layout="wide"
)

hide_menu = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fuzzywithgeo.csv")  # Replace with your actual file path
    # Ensure lat/lng are float
    df["lat"] = df["lat"].astype(float)
    df["lng"] = df["lng"].astype(float)
    return df

data = load_data()

# CLUSTER META (from your analysis)
FPC_SCORE = 0.9227
SILHOUETTE = 0.276
CALINSKI = 1387.4
DAVIES_B = 0.46
N_CLUSTERS = 35

# --------------------------------------------------
# 3. Sidebar – filters
# --------------------------------------------------
st.sidebar.header("🔎 Filter Data")
st.sidebar.markdown("**Dataset Overview**")
st.sidebar.write(f"Total Segments: {len(data):,}")
st.sidebar.write(f"FCM Clusters: {N_CLUSTERS}")
st.sidebar.write(f"Unique Routes: {data['StartLocation'].nunique()} → {data['EndLocation'].nunique()}")

st.sidebar.markdown("---")

# Cluster selection
all_clusters = sorted(data["Cluster_FCM"].unique())
sel_clusters = st.sidebar.multiselect(
    "Select Clusters", 
    all_clusters, 
    default=all_clusters[:10]  # Show first 10 by default
)

# Duration filter
min_dur, max_dur = int(data["SegmentDuration"].min()), int(data["SegmentDuration"].max())
dur_range = st.sidebar.slider(
    "Segment Duration (minutes)", 
    min_dur, max_dur, (min_dur, max_dur)
)

# Weekend filter
weekend_filter = st.sidebar.selectbox(
    "Trip Type",
    ["All", "Weekday Only", "Weekend Only"]
)

# Hour filter
hour_range = st.sidebar.slider(
    "Start Hour Range",
    0, 23, (0, 23)
)

# Apply filters
filtered = data[
    data["Cluster_FCM"].isin(sel_clusters) &
    data["SegmentDuration"].between(*dur_range) &
    data["StartHour"].between(*hour_range)
]

if weekend_filter == "Weekday Only":
    filtered = filtered[filtered["IsWeekend"] == False]
elif weekend_filter == "Weekend Only":
    filtered = filtered[filtered["IsWeekend"] == True]

# --------------------------------------------------
# 4. Main Title and KPIs
# --------------------------------------------------
st.title("🚛 Trip Segment Clustering Analysis")
st.subheader("Fuzzy C-Means Clustering for Unique Transportation Routes")

st.info("""
**📋 Project Context**

This analysis uses **Fuzzy C-Means clustering** to group 122,000 unique trip segments that lacked identical counterparts for direct comparison. 
The resulting **35 clusters** enable fair performance benchmarking and feed into **SQL stored procedures** that predict Load-Drive-Unload times 
for novel trip planning and equitable driver compensation.

**Key Achievement**: FPC Score of 0.9227 indicates highly distinct, well-separated clusters.
""")

# KPIs
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Segments", f"{len(data):,}")
kpi2.metric("FCM Clusters", f"{N_CLUSTERS}")
kpi3.metric("FPC Score", f"{FPC_SCORE:.4f}")
kpi4.metric("Silhouette Score", f"{SILHOUETTE:.3f}")
kpi5.metric("Avg Duration", f"{data['SegmentDuration'].mean():.1f} min")

st.markdown("---")

# --------------------------------------------------
# 5. Tabs
# --------------------------------------------------
tabs = st.tabs([
    "📜 Project Overview",
    "🔬 Model Quality",
    "📊 Cluster Explorer",
    "🗺️ Geographic Heatmap",
    "⏱️ Benchmark Logic",
    "📈 Performance Analysis"
])

# --------------  📜  PROJECT OVERVIEW  ---------------
with tabs[0]:
    st.header("Project at a Glance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Business Problem
        
        **Challenge**: Trip segments lacked identical counterparts for direct performance comparison, making fair driver evaluation difficult.
        
        **Solution**: Fuzzy C-Means clustering groups similar segments to enable:
        - Fair performance benchmarking
        - Equitable driver compensation
        - Novel trip time prediction
        
        ### 🔍 Technical Approach
        
        **Data**: 122,000 unique trip segments from Dijkstra-based route segmentation
        **Method**: Fuzzy C-Means with 35 clusters
        **Features**: Duration, speed, distance, time, location
        **Validation**: FPC = 0.9227, Silhouette = 0.276
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Key Results
        
        **Clustering Quality**:
        - **35 distinct clusters** with minimal overlap
        - **High FPC score (0.9227)** indicates excellent separation
        - **Low intra-cluster variability** enables fair comparisons
        
        **Business Impact**:
        - Powers **3 SQL stored procedures** for time prediction
        - Enables **equitable driver compensation** system
        - Provides **benchmarks for novel routes**
        
        ### 🛠️ Implementation
        
        **Integration**: Cluster IDs feed into production SQL procedures
        **Usage**: `sp_Load`, `sp_Drive`, `sp_Unload` return median times
        **Outcome**: End-to-end trip time estimates for planning
        """)
    
    # Summary insights from the report
    st.subheader("📋 Key Project Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.success("""
        **🏆 Clustering Success**
        - 35 clusters with FPC = 0.9227
        - Average cluster duration: 33 minutes
        - Duration variability: 26 minutes
        - Geographic coherence achieved
        """)
    
    with insight_col2:
        st.info("""
        **🔧 Technical Innovation**
        - Dijkstra-based trip segmentation
        - Fuzzy clustering handles overlaps
        - Multi-dimensional feature analysis
        - Robust validation metrics
        """)
    
    with insight_col3:
        st.warning("""
        **💼 Business Value**
        - Fair driver performance evaluation
        - Novel trip time prediction
        - Compensation equity improvement
        - Scalable benchmarking system
        """)

# --------------  🔬  MODEL QUALITY ----------
with tabs[1]:
    st.header("Clustering Validation & Model Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Final Model Metrics")
        
        # Create metrics table
        metrics_df = pd.DataFrame({
            "Metric": ["Fuzzy Partition Coefficient (FPC)", "Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index"],
            "Score": [FPC_SCORE, SILHOUETTE, CALINSKI, DAVIES_B],
            "Interpretation": ["Excellent (>0.9)", "Good (>0.25)", "High separation", "Low overlap"]
        })
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown("""
        **Interpretation**:
        - **FPC = 0.9227**: Clusters are well-defined with minimal overlap
        - **Silhouette = 0.276**: Good cluster separation and cohesion
        - **High Calinski-Harabasz**: Well-separated clusters
        - **Low Davies-Bouldin**: Compact, distinct clusters
        """)
    
    with col2:
        st.subheader("🎯 Cluster Characteristics")
        
        # Cluster quality analysis
        cluster_stats = filtered.groupby('Cluster_FCM').agg({
            'SegmentDuration': ['mean', 'std', 'count'],
            'AvgSpeed': 'mean',
            'DistanceTraveled_Odometer': 'mean',
            'StartLocation': 'nunique'
        }).round(2)
        
        cluster_stats.columns = ['Avg_Duration', 'Duration_Std', 'Count', 'Avg_Speed', 'Avg_Distance', 'Unique_Locations']
        cluster_stats['Consistency'] = (100 - (cluster_stats['Duration_Std'] / cluster_stats['Avg_Duration'] * 100)).round(1)
        
        # Show top 10 clusters by size
        top_clusters = cluster_stats.nlargest(10, 'Count')[['Count', 'Avg_Duration', 'Consistency']]
        st.dataframe(top_clusters, use_container_width=True)
    
    # Clustering methodology explanation
    st.subheader("🔬 Why Fuzzy C-Means with 35 Clusters?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Fuzzy C-Means Advantages:**
        - Handles overlapping trip characteristics
        - Assigns membership values to multiple clusters
        - Captures transitional or mixed trip types
        - Provides flexibility for multi-dimensional analysis
        """)
    
    with col2:
        st.success("""
        **35 Clusters Selection:**
        - Balanced granularity for meaningful comparisons
        - Reduces intra-cluster variability
        - Maintains interpretability
        - Optimized for business application
        """)
    
    # Cluster validation visualization
    st.subheader("📈 Cluster Quality Validation")
    
    # Create a visualization showing cluster quality metrics
    cluster_quality_metrics = filtered.groupby('Cluster_FCM').agg({
        'SegmentDuration': ['mean', 'std', 'count']
    }).round(2)
    
    cluster_quality_metrics.columns = ['Avg_Duration', 'Duration_Std', 'Count']
    cluster_quality_metrics['Consistency_Score'] = (100 - (cluster_quality_metrics['Duration_Std'] / cluster_quality_metrics['Avg_Duration'] * 100)).round(1)
    cluster_quality_metrics = cluster_quality_metrics.reset_index()
    
    fig_quality = px.scatter(
        cluster_quality_metrics,
        x='Count',
        y='Consistency_Score',
        size='Avg_Duration',
        color='Cluster_FCM',
        title="Cluster Quality: Size vs Consistency (Bubble size = Avg Duration)",
        labels={'Count': 'Number of Segments', 'Consistency_Score': 'Consistency Score (%)'}
    )
    st.plotly_chart(fig_quality, use_container_width=True)

# --------------  📊  CLUSTER EXPLORER  ------
with tabs[2]:
    st.header("Interactive Cluster Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster size distribution
        cluster_counts = filtered["Cluster_FCM"].value_counts().sort_index()
        fig_bar = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={"x": "Cluster ID", "y": "Number of Segments"},
            title="Segment Count per Cluster"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Duration distribution by cluster
        fig_violin = px.violin(
            filtered, 
            x="Cluster_FCM", 
            y="SegmentDuration",
            box=True, 
            points=False,
            title="Duration Distribution by Cluster"
        )
        fig_violin.update_xaxes(type="category")
        st.plotly_chart(fig_violin, use_container_width=True)
    
    # Speed vs Distance analysis
    st.subheader("🚗 Speed vs Distance Analysis")
    
    fig_scatter = px.scatter(
        filtered, 
        x="DistanceTraveled_Odometer", 
        y="AvgSpeed",
        color="Cluster_FCM", 
        size="SegmentDuration",
        opacity=0.6,
        title="Distance vs Average Speed (Size = Duration)",
        labels={"DistanceTraveled_Odometer": "Distance (Odometer)", "AvgSpeed": "Average Speed (mph)"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Time-based analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Hour distribution
        hour_dist = filtered.groupby(['StartHour', 'Cluster_FCM']).size().reset_index(name='Count')
        fig_hour = px.bar(
            hour_dist, 
            x='StartHour', 
            y='Count', 
            color='Cluster_FCM',
            title="Trip Start Time Distribution by Cluster"
        )
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col2:
        # Weekend vs Weekday
        weekend_dist = filtered.groupby(['IsWeekend', 'Cluster_FCM']).size().reset_index(name='Count')
        fig_weekend = px.bar(
            weekend_dist, 
            x='IsWeekend', 
            y='Count', 
            color='Cluster_FCM',
            title="Weekend vs Weekday Distribution"
        )
        st.plotly_chart(fig_weekend, use_container_width=True)

# --------------  🗺️  GEOGRAPHIC HEATMAP  ----------
# --------------  📊  TEMPORAL & OPERATIONAL ANALYSIS  ----------
with tabs[3]:
    st.header("Temporal & Operational Pattern Analysis")
    
    # Time-based clustering patterns
    st.subheader("🕐 Temporal Clustering Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster activity by hour of day
        hourly_cluster = filtered.groupby(['StartHour', 'Cluster_FCM']).size().reset_index(name='Count')
        fig_hourly = px.line(
            hourly_cluster, 
            x='StartHour', 
            y='Count', 
            color='Cluster_FCM',
            title="Cluster Activity Throughout the Day",
            labels={'StartHour': 'Hour of Day', 'Count': 'Number of Segments'}
        )
        fig_hourly.update_layout(showlegend=False)
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Weekend vs Weekday cluster distribution
        weekend_cluster = filtered.groupby(['Cluster_FCM', 'IsWeekend']).size().reset_index(name='Count')
        weekend_cluster['Day_Type'] = weekend_cluster['IsWeekend'].map({True: 'Weekend', False: 'Weekday'})
        
        fig_weekend = px.bar(
            weekend_cluster,
            x='Cluster_FCM',
            y='Count',
            color='Day_Type',
            title="Weekday vs Weekend Distribution by Cluster",
            labels={'Cluster_FCM': 'Cluster ID', 'Count': 'Number of Segments'}
        )
        st.plotly_chart(fig_weekend, use_container_width=True)
    
    # Operational efficiency analysis
    st.subheader("⚡ Operational Efficiency by Cluster")
    
    # Calculate efficiency metrics
    efficiency_metrics = filtered.groupby('Cluster_FCM').agg({
        'SegmentDuration': ['mean', 'std', 'count'],
        'AvgSpeed': ['mean', 'std'],
        'DistanceTraveled_Odometer': ['mean', 'std'],
        'IsWeekend': lambda x: (x.sum() / len(x)) * 100
    }).round(2)
    
    efficiency_metrics.columns = ['Avg_Duration', 'Duration_Std', 'Count', 'Avg_Speed', 'Speed_Std', 'Avg_Distance', 'Distance_Std', 'Weekend_Pct']
    efficiency_metrics['Speed_Efficiency'] = (efficiency_metrics['Avg_Speed'] / efficiency_metrics['Avg_Duration']).round(3)
    efficiency_metrics['Consistency_Score'] = (100 - (efficiency_metrics['Duration_Std'] / efficiency_metrics['Avg_Duration'] * 100)).round(1)
    
    # Efficiency scatter plot
    fig_efficiency = px.scatter(
        efficiency_metrics.reset_index(),
        x='Avg_Duration',
        y='Speed_Efficiency',
        size='Count',
        color='Consistency_Score',
        hover_data=['Cluster_FCM', 'Avg_Speed', 'Avg_Distance'],
        title="Cluster Efficiency: Duration vs Speed Efficiency (Size = Count, Color = Consistency)",
        labels={
            'Avg_Duration': 'Average Duration (minutes)',
            'Speed_Efficiency': 'Speed Efficiency (mph/min)',
            'Consistency_Score': 'Consistency Score (%)'
        }
    )
    st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Driver performance insights
    st.subheader("👨‍💼 Driver Performance Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most consistent clusters (low variability)
        consistent_clusters = efficiency_metrics.nlargest(8, 'Consistency_Score')[['Count', 'Avg_Duration', 'Consistency_Score']]
        st.success("**✅ Most Consistent Clusters (Best for Benchmarking)**")
        st.dataframe(consistent_clusters, use_container_width=True)
        
        st.caption("These clusters show low duration variability, making them ideal for setting performance benchmarks.")
    
    with col2:
        # High variability clusters (need attention)
        variable_clusters = efficiency_metrics.nsmallest(8, 'Consistency_Score')[['Count', 'Avg_Duration', 'Consistency_Score']]
        st.warning("**⚠️ High Variability Clusters (Need Review)**")
        st.dataframe(variable_clusters, use_container_width=True)
        
        st.caption("These clusters show high duration variability, indicating potential issues or diverse conditions.")
    
    # Cluster characteristics heatmap
    st.subheader("🔥 Cluster Characteristics Heatmap")
    
    # Normalize metrics for heatmap
    heatmap_data = efficiency_metrics[['Avg_Duration', 'Avg_Speed', 'Avg_Distance', 'Weekend_Pct', 'Consistency_Score']].copy()
    
    # Normalize each column to 0-1 scale
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    heatmap_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        columns=heatmap_data.columns,
        index=heatmap_data.index
    )
    
    fig_heatmap = px.imshow(
        heatmap_normalized.T,
        x=heatmap_normalized.index,
        y=heatmap_normalized.columns,
        title="Normalized Cluster Characteristics (0=Min, 1=Max)",
        labels={'x': 'Cluster ID', 'y': 'Metric', 'color': 'Normalized Value'},
        color_continuous_scale="RdYlBu_r"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Operational insights
    st.subheader("💡 Key Operational Insights")
    
    # Calculate some insights
    peak_hour = filtered.groupby('StartHour').size().idxmax()
    most_active_cluster = filtered['Cluster_FCM'].value_counts().index[0]
    weekend_heavy_clusters = efficiency_metrics[efficiency_metrics['Weekend_Pct'] > 30].shape[0]
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.info(f"""
        **⏰ Peak Activity**
        - Peak hour: {peak_hour}:00
        - Most active cluster: {most_active_cluster}
        - {len(efficiency_metrics)} total clusters analyzed
        """)
    
    with insight_col2:
        st.success(f"""
        **📊 Performance Distribution**
        - Avg consistency: {efficiency_metrics['Consistency_Score'].mean():.1f}%
        - Speed range: {efficiency_metrics['Avg_Speed'].min():.1f}-{efficiency_metrics['Avg_Speed'].max():.1f} mph
        - Duration range: {efficiency_metrics['Avg_Duration'].min():.1f}-{efficiency_metrics['Avg_Duration'].max():.1f} min
        """)
    
    with insight_col3:
        st.warning(f"""
        **🔍 Areas for Improvement**
        - {weekend_heavy_clusters} clusters are weekend-heavy (>30%)
        - Variability range: {efficiency_metrics['Consistency_Score'].min():.1f}%-{efficiency_metrics['Consistency_Score'].max():.1f}%
        - Focus on low-consistency clusters
        """)


# --------------  ⏱️  BENCHMARK LOGIC -------
with tabs[4]:
    st.header("From Clustering to SQL Benchmarking")
    
    st.markdown("""
    ### 🔄 Integration Workflow
    
    The FCM clustering results integrate with Groendyke's operational system through three SQL stored procedures:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📦 sp_Load**
        
        **Input**: Event type, commodity, location, time window
        **Process**: Query historical load events
        **Output**: Median load time (minutes)
        **Uses**: Location clustering for sparse data
        """)
    
    with col2:
        st.success("""
        **🚛 sp_Drive**
        
        **Input**: Start/end location, day, time, **cluster ID**
        **Process**: Chain segments using Dijkstra's algorithm
        **Output**: Total drive time estimate
        **Uses**: FCM cluster for segment benchmarking
        """)
    
    with col3:
        st.warning("""
        **📤 sp_Unload**
        
        **Input**: Event type, commodity, location, time window
        **Process**: Query historical unload events
        **Output**: Median unload time (minutes)
        **Uses**: Location clustering for sparse data
        """)
    
    st.markdown("""
    ### 🔗 Process Flow
    
    ```
    1. New Trip Request → Extract features (start, end, time, etc.)
    2. Feature Matching → Assign to FCM cluster based on similarity
    3. SQL Procedures → Query historical data using cluster ID
    4. Benchmark Calculation → Sum load + drive + unload times
    5. Output → Total estimated trip time for planning
    ```
    
    ### 🎯 Key Benefits
    
    - **Fair Comparison**: Clusters enable comparison of similar trip types
    - **Novel Trip Handling**: Predict times for routes never driven before
    - **Scalable System**: Framework handles growing dataset
    - **Production Ready**: Integrated with existing Groendyke architecture
    """)
    
    # Show example cluster characteristics
    st.subheader("📊 Example: Cluster Characteristics")
    
    # Select a representative cluster
    example_cluster = filtered[filtered['Cluster_FCM'] == filtered['Cluster_FCM'].mode()[0]]
    
    cluster_summary = {
        "Metric": ["Average Duration", "Average Speed", "Average Distance", "Most Common Start Hour", "Weekend Percentage"],
        "Value": [
            f"{example_cluster['SegmentDuration'].mean():.1f} minutes",
            f"{example_cluster['AvgSpeed'].mean():.1f} mph",
            f"{example_cluster['DistanceTraveled_Odometer'].mean():.1f} miles",
            f"{example_cluster['StartHour'].mode()[0]}:00",
            f"{(example_cluster['IsWeekend'].sum() / len(example_cluster) * 100):.1f}%"
        ]
    }
    
    st.table(pd.DataFrame(cluster_summary))

# --------------  📈  PERFORMANCE ANALYSIS ----------
with tabs[5]:
    st.header("Cluster Performance & Efficiency Analysis")
    
    # Calculate performance metrics
    cluster_performance = filtered.groupby('Cluster_FCM').agg({
        'SegmentDuration': ['mean', 'std', 'count'],
        'AvgSpeed': 'mean',
        'DistanceTraveled_Odometer': 'mean',
        'IsWeekend': lambda x: (x.sum() / len(x)) * 100,
        'StartLocation': 'nunique',
        'EndLocation': 'nunique'
    }).round(2)
    
    cluster_performance.columns = ['Avg_Duration', 'Duration_Std', 'Count', 'Avg_Speed', 'Avg_Distance', 'Weekend_Pct', 'Unique_Starts', 'Unique_Ends']
    cluster_performance['Efficiency_Score'] = (cluster_performance['Avg_Speed'] / cluster_performance['Avg_Duration'] * 100).round(2)
    cluster_performance['Consistency_Score'] = (100 - (cluster_performance['Duration_Std'] / cluster_performance['Avg_Duration'] * 100)).round(1)
    
    # Top performing clusters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Most Efficient Clusters")
        top_efficient = cluster_performance.nlargest(5, 'Efficiency_Score')[['Count', 'Avg_Duration', 'Avg_Speed', 'Efficiency_Score']]
        st.dataframe(top_efficient, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Most Consistent Clusters")
        top_consistent = cluster_performance.nlargest(5, 'Consistency_Score')[['Count', 'Avg_Duration', 'Duration_Std', 'Consistency_Score']]
        st.dataframe(top_consistent, use_container_width=True)
    
    # Performance visualization
    fig_performance = px.scatter(
        cluster_performance.reset_index(),
        x='Avg_Duration',
        y='Avg_Speed',
        size='Count',
        color='Efficiency_Score',
        hover_data=['Cluster_FCM', 'Consistency_Score'],
        title="Cluster Performance: Duration vs Speed (Size = Count, Color = Efficiency)",
        labels={'Avg_Duration': 'Average Duration (minutes)', 'Avg_Speed': 'Average Speed (mph)'}
    )
    st.plotly_chart(fig_performance, use_container_width=True)
    
    # Route diversity analysis
    st.subheader("🛣️ Route Diversity Analysis")
    
    fig_diversity = px.scatter(
        cluster_performance.reset_index(),
        x='Unique_Starts',
        y='Unique_Ends',
        size='Count',
        color='Cluster_FCM',
        title="Route Diversity: Unique Start vs End Locations",
        labels={'Unique_Starts': 'Unique Start Locations', 'Unique_Ends': 'Unique End Locations'}
    )
    st.plotly_chart(fig_diversity, use_container_width=True)

# --------------------------------------------------
# 6. Data Export Section
# --------------------------------------------------
st.markdown("---")
st.subheader("📊 Data Export")

# Download filtered data
csv = filtered.to_csv(index=False).encode()
st.download_button(
    label="📥 Download Filtered Data (CSV)",
    data=csv,
    file_name='filtered_trip_segments.csv',
    mime='text/csv'
)

# Summary statistics
st.subheader("📈 Current Filter Summary")
summary_stats = {
    "Metric": [
        "Total Segments",
        "Clusters Shown",
        "Average Duration",
        "Average Speed",
        "Average Distance",
        "Weekend Percentage"
    ],
    "Value": [
        f"{len(filtered):,}",
        f"{filtered['Cluster_FCM'].nunique()}",
        f"{filtered['SegmentDuration'].mean():.1f} minutes",
        f"{filtered['AvgSpeed'].mean():.1f} mph",
        f"{filtered['DistanceTraveled_Odometer'].mean():.1f} miles",
        f"{(filtered['IsWeekend'].sum() / len(filtered) * 100):.1f}%"
    ]
}

col1, col2 = st.columns(2)
with col1:
    st.table(pd.DataFrame(summary_stats))

with col2:
    st.info("""
    **💡 Key Takeaways**
    
    - **35 FCM clusters** successfully group diverse trip segments
    - **High FPC score (0.9227)** demonstrates excellent clustering quality
    - **Production integration** through SQL stored procedures
    - **Fair benchmarking** enables equitable driver compensation
    - **Novel trip prediction** supports operational planning
    """)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <center>
    <b>Trip Segment Clustering Analysis</b> | Fuzzy C-Means Implementation<br>
    Data Science Portfolio Project | 2025
    </center>
    """,
    unsafe_allow_html=True
)

