import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="🏃‍♂️ Dad's Running Performance Dashboard",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .distance-header {
        color: #2E86AB;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from Google Sheets"""
    # Google Sheets CSV export URL
    sheet_id = "1CUM-P3wB2zxHrbmw1JM7vrxtqhtnsasy34NqaIkke_0"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    
    try:
        # Add headers to mimic browser request for better compatibility
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if we got CSV data
        content = response.text
        if 'DOCTYPE html' in content or '<html' in content:
            st.error("⚠️ Google Sheets permission issue. Please ensure the sheet is publicly accessible!")
            st.info("💡 Go to your Google Sheet → Share → 'Anyone with the link can view'")
            return None
            
        df = pd.read_csv(StringIO(content))
        
        # Validate that we have the expected columns
        expected_cols = ['100m_Time', '100m_Date', '200m_Time', '200m_Date', '300m_Time', '300m_Date', '400m_Time', '400m_Date', '500m_Time', '500m_Date']
        if not any(col in df.columns for col in expected_cols):
            st.error("❌ Data format issue. Expected columns not found!")
            st.info("💡 Check that your Google Sheet has columns like '100m_Time', '100m_Date', etc.")
            return None
            
        return df
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout loading data. Please try refreshing the page.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Network error loading data: {e}")
        st.info("💡 Check your internet connection and Google Sheets permissions.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def parse_time_to_seconds(time_str):
    """Convert time string to total seconds"""
    if pd.isna(time_str) or time_str == '':
        return None
    
    time_str = str(time_str).strip()
    
    try:
        # Handle different time formats
        if '.' in time_str and ':' not in time_str:
            # Format like "1.10.0" (minutes.seconds.milliseconds)
            parts = time_str.split('.')
            if len(parts) == 3:
                minutes = int(parts[0])
                seconds = int(parts[1])
                milliseconds = int(parts[2])
                return minutes * 60 + seconds + milliseconds / 10
            elif len(parts) == 2:
                seconds = int(parts[0])
                milliseconds = int(parts[1])
                return seconds + milliseconds / 10
        elif ':' in time_str:
            # Format like "1:10.5"
            parts = time_str.split(':')
            minutes = int(parts[0])
            sec_parts = parts[1].split('.')
            seconds = int(sec_parts[0])
            milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
            return minutes * 60 + seconds + milliseconds / 10
        else:
            # Simple seconds format
            return float(time_str)
    except:
        return None

def parse_date(date_str):
    """Parse date string to datetime"""
    if pd.isna(date_str) or date_str == '':
        return None
    
    try:
        # Handle format like "27-12-24"
        date_str = str(date_str).strip()
        if '-' in date_str:
            parts = date_str.split('-')
            if len(parts) == 3:
                day, month, year = parts
                # Convert 2-digit year to 4-digit
                if len(year) == 2:
                    year = '20' + year
                return pd.to_datetime(f"{year}-{month}-{day}")
    except:
        pass
    
    return None

def process_data(df):
    """Process raw data into structured format"""
    distances = ['100m', '200m', '300m', '400m', '500m']
    processed_data = {}
    
    for distance in distances:
        time_col = f"{distance}_Time"
        date_col = f"{distance}_Date"
        
        if time_col in df.columns and date_col in df.columns:
            # Get non-null rows
            mask = df[time_col].notna() & df[date_col].notna()
            times = df.loc[mask, time_col]
            dates = df.loc[mask, date_col]
            
            # Process times and dates
            processed_times = [parse_time_to_seconds(t) for t in times]
            processed_dates = [parse_date(d) for d in dates]
            
            # Filter out None values
            valid_data = [(t, d) for t, d in zip(processed_times, processed_dates) 
                         if t is not None and d is not None]
            
            if valid_data:
                times_clean, dates_clean = zip(*valid_data)
                processed_data[distance] = pd.DataFrame({
                    'time_seconds': times_clean,
                    'date': dates_clean,
                    'distance': distance
                }).sort_values('date')
    
    return processed_data

def format_time(seconds):
    """Format seconds back to readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:04.1f}"

def create_performance_chart(data, distance):
    """Create performance over time chart"""
    if distance not in data or data[distance].empty:
        return None
    
    df = data[distance].copy()
    
    # Calculate trend
    if len(df) > 1:
        z = np.polyfit(range(len(df)), df['time_seconds'], 1)
        trend = np.poly1d(z)
        df['trend'] = trend(range(len(df)))
    
    fig = go.Figure()
    
    # Add actual times
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['time_seconds'],
        mode='markers+lines',
        name='Actual Times',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=8, color='#FF6B35')
    ))
    
    # Add trend line
    if len(df) > 1:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='#2E86AB', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f"{distance} Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Time (seconds)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_seasonal_analysis(data):
    """Create comprehensive seasonal performance analysis"""
    all_data = []
    for distance, df in data.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['month'] = df_copy['date'].dt.month
            df_copy['month_name'] = df_copy['date'].dt.strftime('%B')
            df_copy['season'] = df_copy['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            all_data.append(df_copy)
    
    if not all_data:
        return None, None, None
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # 1. Seasonal comparison by distance
    seasonal_stats = combined.groupby(['distance', 'season']).agg({
        'time_seconds': ['mean', 'min', 'count', 'std']
    }).round(2)
    seasonal_stats.columns = ['avg_time', 'best_time', 'run_count', 'consistency']
    seasonal_stats = seasonal_stats.reset_index()
    
    # Create subplots for each distance
    distances = sorted(combined['distance'].unique())
    fig_seasonal = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{dist} Performance by Season" for dist in distances] + ["Overall Seasonal Trends"],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"colspan": 1}]]
    )
    
    colors = ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5', '#2E86AB']
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    # Plot each distance separately
    for i, distance in enumerate(distances):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        distance_data = seasonal_stats[seasonal_stats['distance'] == distance]
        
        # Only plot if we have data for multiple seasons
        if len(distance_data) > 1:
            fig_seasonal.add_trace(
                go.Bar(
                    x=distance_data['season'],
                    y=distance_data['avg_time'],
                    name=f"{distance} Avg",
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                    text=[f"{t:.1f}s" for t in distance_data['avg_time']],
                    textposition='auto'
                ),
                row=row, col=col
            )
    
    # Overall seasonal performance (normalized scores)
    if len(distances) > 1:
        seasonal_normalized = create_normalized_seasonal_scores(seasonal_stats, distances)
        if seasonal_normalized is not None:
            fig_seasonal.add_trace(
                go.Bar(
                    x=seasonal_normalized['season'],
                    y=seasonal_normalized['combined_score'],
                    name="Combined Performance",
                    marker_color='#764ba2',
                    showlegend=False,
                    text=[f"{s:.1f}" for s in seasonal_normalized['combined_score']],
                    textposition='auto'
                ),
                row=2, col=3
            )
    
    fig_seasonal.update_layout(
        title="Seasonal Performance Analysis by Distance",
        template='plotly_white',
        height=600
    )
    
    # 2. Monthly detailed analysis
    monthly_stats = combined.groupby(['distance', 'month_name']).agg({
        'time_seconds': ['mean', 'min', 'count']
    }).round(2)
    monthly_stats.columns = ['avg_time', 'best_time', 'run_count']
    monthly_stats = monthly_stats.reset_index()
    
    # 3. Best performing periods analysis
    best_periods = analyze_best_periods(combined)
    
    return fig_seasonal, monthly_stats, best_periods

def create_normalized_seasonal_scores(seasonal_stats, distances):
    """Create normalized performance scores for combined seasonal analysis"""
    try:
        normalized_data = []
        
        for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
            season_data = seasonal_stats[seasonal_stats['season'] == season]
            if len(season_data) == 0:
                continue
                
            # Calculate normalized scores (lower time = higher score)
            scores = []
            for distance in distances:
                dist_data = season_data[season_data['distance'] == distance]
                if len(dist_data) > 0:
                    time_val = dist_data['avg_time'].iloc[0]
                    # Normalize based on distance expectations
                    if distance == '100m':
                        score = max(0, 100 - (time_val - 18) * 10)  # Expected ~18-20s
                    elif distance == '200m':
                        score = max(0, 100 - (time_val - 40) * 5)   # Expected ~40-45s
                    elif distance == '300m':
                        score = max(0, 100 - (time_val - 70) * 3)   # Expected ~70-80s
                    elif distance == '400m':
                        score = max(0, 100 - (time_val - 100) * 2)  # Expected ~100-110s
                    elif distance == '500m':
                        score = max(0, 100 - (time_val - 130) * 1.5) # Expected ~130-150s
                    else:
                        score = 50  # Default
                    scores.append(score)
            
            if scores:
                combined_score = np.mean(scores)
                normalized_data.append({
                    'season': season,
                    'combined_score': combined_score,
                    'distance_count': len(scores)
                })
        
        return pd.DataFrame(normalized_data) if normalized_data else None
    except:
        return None

def analyze_best_periods(combined_data):
    """Analyze best performing periods"""
    try:
        # Monthly analysis
        monthly_performance = combined_data.groupby(['month', 'distance']).agg({
            'time_seconds': ['mean', 'count']
        }).round(2)
        monthly_performance.columns = ['avg_time', 'run_count']
        monthly_performance = monthly_performance.reset_index()
        
        # Find best month for each distance
        best_months = {}
        for distance in combined_data['distance'].unique():
            dist_data = monthly_performance[monthly_performance['distance'] == distance]
            if len(dist_data) > 0:
                best_month_idx = dist_data['avg_time'].idxmin()
                best_month_data = dist_data.loc[best_month_idx]
                best_months[distance] = {
                    'month': best_month_data['month'],
                    'avg_time': best_month_data['avg_time'],
                    'run_count': best_month_data['run_count']
                }
        
        # Seasonal analysis
        seasonal_performance = combined_data.groupby(['season', 'distance']).agg({
            'time_seconds': ['mean', 'count']
        }).round(2)
        seasonal_performance.columns = ['avg_time', 'run_count']
        seasonal_performance = seasonal_performance.reset_index()
        
        return {
            'best_months': best_months,
            'monthly_data': monthly_performance,
            'seasonal_data': seasonal_performance
        }
    except:
        return None

def create_monthly_heatmap(data):
    """Create a heatmap showing performance across months and distances"""
    all_data = []
    for distance, df in data.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['month'] = df_copy['date'].dt.month
            df_copy['month_name'] = df_copy['date'].dt.strftime('%B')
            all_data.append(df_copy)
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Create pivot table for heatmap
    heatmap_data = combined.groupby(['distance', 'month_name'])['time_seconds'].mean().unstack(fill_value=0)
    
    # Order months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Reorder columns to match month order
    available_months = [month for month in month_order if month in heatmap_data.columns]
    heatmap_data = heatmap_data[available_months]
    
    # Replace 0s with NaN for better visualization
    heatmap_data = heatmap_data.replace(0, np.nan)
    
    fig = px.imshow(
        heatmap_data,
        title="Performance Heatmap: Average Times by Month and Distance",
        labels={'x': 'Month', 'y': 'Distance', 'color': 'Avg Time (seconds)'},
        color_continuous_scale='RdYlBu_r',  # Red = slower, Blue = faster
        aspect='auto'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400
    )
    
    return fig

def create_improvement_by_season_chart(data):
    """Show improvement trends by season"""
    all_data = []
    for distance, df in data.items():
        if len(df) >= 3:  # Need enough data points
            df_copy = df.copy()
            df_copy['month'] = df_copy['date'].dt.month
            df_copy['season'] = df_copy['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            
            # Calculate improvement within each season
            seasonal_improvements = []
            for season in df_copy['season'].unique():
                season_data = df_copy[df_copy['season'] == season].sort_values('date')
                if len(season_data) >= 2:
                    first_time = season_data.iloc[0]['time_seconds']
                    best_time = season_data['time_seconds'].min()
                    improvement = ((first_time - best_time) / first_time) * 100
                    seasonal_improvements.append({
                        'distance': distance,
                        'season': season,
                        'improvement': improvement,
                        'runs': len(season_data)
                    })
            
            all_data.extend(seasonal_improvements)
    
    if not all_data:
        return None
    
    improvement_df = pd.DataFrame(all_data)
    
    fig = px.bar(
        improvement_df,
        x='season',
        y='improvement',
        color='distance',
        title="Performance Improvement by Season (%)",
        labels={'improvement': 'Improvement (%)', 'season': 'Season'},
        category_orders={'season': ['Spring', 'Summer', 'Autumn', 'Winter']}
    )
    
    fig.update_layout(template='plotly_white')
    return fig

def create_improvement_chart(data):
    """Create improvement percentage chart"""
    improvements = {}
    
    for distance, df in data.items():
        if len(df) >= 2:
            first_time = df.iloc[0]['time_seconds']
            best_time = df['time_seconds'].min()
            improvement = ((first_time - best_time) / first_time) * 100
            improvements[distance] = improvement
    
    if not improvements:
        return None
    
    distances = list(improvements.keys())
    values = list(improvements.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=distances,
            y=values,
            marker_color=['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5', '#2E86AB'],
            text=[f"{v:.1f}%" for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Performance Improvement by Distance",
        xaxis_title="Distance",
        yaxis_title="Improvement (%)",
        template='plotly_white'
    )
    
    return fig

def create_consistency_chart(data):
    """Create consistency analysis (coefficient of variation)"""
    consistency = {}
    
    for distance, df in data.items():
        if len(df) >= 3:
            cv = (df['time_seconds'].std() / df['time_seconds'].mean()) * 100
            consistency[distance] = cv
    
    if not consistency:
        return None
    
    distances = list(consistency.keys())
    values = list(consistency.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=distances,
            y=values,
            marker_color=['#764ba2', '#667eea', '#f093fb', '#f5576c', '#4facfe'],
            text=[f"{v:.1f}%" for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Performance Consistency (Lower = More Consistent)",
        xaxis_title="Distance",
        yaxis_title="Coefficient of Variation (%)",
        template='plotly_white'
    )
    
    return fig

def create_distribution_chart(data, distance):
    """Create time distribution histogram"""
    if distance not in data or data[distance].empty:
        return None
    
    df = data[distance]
    
    fig = px.histogram(
        df,
        x='time_seconds',
        nbins=min(10, len(df)),
        title=f"{distance} Time Distribution",
        labels={'time_seconds': 'Time (seconds)', 'count': 'Frequency'}
    )
    
    fig.update_layout(template='plotly_white')
    return fig

def create_progress_radar(data):
    """Create radar chart for multi-distance performance"""
    if len(data) < 2:
        return None
    
    # Calculate performance scores (inverse of time, normalized)
    scores = {}
    for distance, df in data.items():
        if not df.empty:
            best_time = df['time_seconds'].min()
            # Normalize score (arbitrary scaling for radar chart)
            if distance == '100m':
                scores[distance] = max(0, 100 - (best_time - 15) * 5)
            elif distance == '200m':
                scores[distance] = max(0, 100 - (best_time - 35) * 2)
            elif distance == '300m':
                scores[distance] = max(0, 100 - (best_time - 60) * 1.5)
            elif distance == '400m':
                scores[distance] = max(0, 100 - (best_time - 90) * 1)
            elif distance == '500m':
                scores[distance] = max(0, 100 - (best_time - 120) * 0.8)
    
    if len(scores) < 3:
        return None
    
    distances = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=distances,
        fill='toself',
        name='Performance Score',
        line_color='#FF6B35'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="Multi-Distance Performance Radar",
        template='plotly_white'
    )
    
    return fig

# Main app
def main():
    st.markdown('<div class="main-header">🏃‍♂️ Dad\'s Running Performance Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading running data..."):
        df = load_data()
    
    if df is None:
        st.error("Could not load data. Please check the Google Sheets link.")
        return
    
    # Process data
    data = process_data(df)
    
    if not data:
        st.error("No valid data found in the spreadsheet.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("🏃‍♂️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a view:",
        ["📊 Overview", "📈 Individual Distances", "🌟 Performance Analysis", "📅 Seasonal Trends", "➕ Add New Data"]
    )
    
    if page == "📊 Overview":
        show_overview(data)
    elif page == "📈 Individual Distances":
        show_individual_distances(data)
    elif page == "🌟 Performance Analysis":
        show_performance_analysis(data)
    elif page == "📅 Seasonal Trends":
        show_seasonal_trends(data)
    elif page == "➕ Add New Data":
        show_data_entry()

def show_overview(data):
    """Show overview dashboard"""
    st.header("📊 Performance Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_runs = sum(len(df) for df in data.values())
    distances_tracked = len(data)
    
    with col1:
        st.metric("Total Runs", total_runs)
    
    with col2:
        st.metric("Distances Tracked", distances_tracked)
    
    # Best times
    best_times = {}
    for distance, df in data.items():
        if not df.empty:
            best_times[distance] = df['time_seconds'].min()
    
    with col3:
        if best_times:
            fastest_distance = min(best_times.keys(), key=lambda x: best_times[x] / (int(x[:-1]) / 100))
            st.metric("Strongest Distance", fastest_distance)
    
    with col4:
        if len(data) >= 2:
            avg_improvement = np.mean([
                ((df.iloc[0]['time_seconds'] - df['time_seconds'].min()) / df.iloc[0]['time_seconds']) * 100
                for df in data.values() if len(df) >= 2
            ])
            st.metric("Avg Improvement", f"{avg_improvement:.1f}%")
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        improvement_chart = create_improvement_chart(data)
        if improvement_chart:
            st.plotly_chart(improvement_chart, use_container_width=True)
    
    with col2:
        consistency_chart = create_consistency_chart(data)
        if consistency_chart:
            st.plotly_chart(consistency_chart, use_container_width=True)
    
    # Radar chart
    radar_chart = create_progress_radar(data)
    if radar_chart:
        st.plotly_chart(radar_chart, use_container_width=True)
    
    # Personal records table
    st.subheader("🏆 Personal Records")
    pr_data = []
    for distance, df in data.items():
        if not df.empty:
            best_idx = df['time_seconds'].idxmin()
            best_time = df.loc[best_idx, 'time_seconds']
            best_date = df.loc[best_idx, 'date']
            pr_data.append({
                'Distance': distance,
                'Personal Record': format_time(best_time),
                'Date Achieved': best_date.strftime('%d-%m-%Y')
            })
    
    if pr_data:
        st.dataframe(pd.DataFrame(pr_data), use_container_width=True)

def show_individual_distances(data):
    """Show individual distance analysis"""
    st.header("📈 Individual Distance Analysis")
    
    distance = st.selectbox("Select Distance", list(data.keys()))
    
    if distance in data and not data[distance].empty:
        df = data[distance]
        
        # Key stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Runs", len(df))
        
        with col2:
            st.metric("Personal Record", format_time(df['time_seconds'].min()))
        
        with col3:
            st.metric("Average Time", format_time(df['time_seconds'].mean()))
        
        with col4:
            if len(df) >= 2:
                improvement = ((df.iloc[0]['time_seconds'] - df['time_seconds'].min()) / df.iloc[0]['time_seconds']) * 100
                st.metric("Total Improvement", f"{improvement:.1f}%")
        
        # Performance chart
        perf_chart = create_performance_chart(data, distance)
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
        
        # Distribution chart
        dist_chart = create_distribution_chart(data, distance)
        if dist_chart:
            st.plotly_chart(dist_chart, use_container_width=True)
        
        # Recent runs table
        st.subheader(f"Recent {distance} Runs")
        recent_df = df.tail(10).copy()
        recent_df['formatted_time'] = recent_df['time_seconds'].apply(format_time)
        recent_df['formatted_date'] = recent_df['date'].dt.strftime('%d-%m-%Y')
        
        display_df = recent_df[['formatted_date', 'formatted_time']].copy()
        display_df.columns = ['Date', 'Time']
        st.dataframe(display_df.iloc[::-1], use_container_width=True)

def show_performance_analysis(data):
    """Show performance analysis"""
    st.header("🌟 Advanced Performance Analysis")
    
    # Monthly performance trends
    st.subheader("📅 Monthly Performance Trends")
    
    all_data = []
    for distance, df in data.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['month'] = df_copy['date'].dt.to_period('M')
            df_copy['distance'] = distance
            all_data.append(df_copy)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        monthly_avg = combined.groupby(['distance', 'month'])['time_seconds'].mean().reset_index()
        monthly_avg['month_str'] = monthly_avg['month'].astype(str)
        
        fig = px.line(
            monthly_avg,
            x='month_str',
            y='time_seconds',
            color='distance',
            title="Monthly Average Performance",
            labels={'time_seconds': 'Average Time (seconds)', 'month_str': 'Month'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance correlation matrix
    if len(data) >= 2:
        st.subheader("🔗 Distance Performance Correlations")
        
        # Create correlation data
        correlation_data = {}
        for distance, df in data.items():
            if not df.empty:
                # Use recent performance (last 5 runs or all if less than 5)
                recent_times = df.tail(5)['time_seconds'].values
                correlation_data[distance] = recent_times
        
        # Pad arrays to same length for correlation
        max_len = max(len(v) for v in correlation_data.values()) if correlation_data else 0
        
        if max_len >= 2:
            padded_data = {}
            for distance, times in correlation_data.items():
                # Pad with mean if needed
                if len(times) < max_len:
                    mean_time = np.mean(times)
                    padded_times = np.concatenate([times, [mean_time] * (max_len - len(times))])
                else:
                    padded_times = times[-max_len:]  # Take last max_len values
                padded_data[distance] = padded_times
            
            corr_df = pd.DataFrame(padded_data)
            correlation_matrix = corr_df.corr()
            
            fig = px.imshow(
                correlation_matrix,
                title="Performance Correlation Between Distances",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)

def show_seasonal_trends(data):
    """Show enhanced seasonal trends analysis"""
    st.header("📅 Seasonal Performance Analysis")
    
    # Enhanced seasonal analysis
    seasonal_chart, monthly_stats, best_periods = create_seasonal_analysis(data)
    
    if seasonal_chart:
        st.plotly_chart(seasonal_chart, use_container_width=True)
        
        # Performance insights
        if best_periods and 'best_months' in best_periods:
            st.subheader("🏆 Peak Performance Periods")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌟 Best Months by Distance:**")
                for distance, info in best_periods['best_months'].items():
                    month_name = pd.to_datetime(f"2024-{info['month']}-01").strftime('%B')
                    st.write(f"• **{distance}**: {month_name} ({format_time(info['avg_time'])} avg)")
            
            with col2:
                # Find overall best season
                if 'seasonal_data' in best_periods:
                    seasonal_avg = best_periods['seasonal_data'].groupby('season')['avg_time'].mean()
                    best_season = seasonal_avg.idxmin()
                    best_season_time = seasonal_avg.min()
                    
                    st.markdown("**🎯 Performance Summary:**")
                    st.write(f"• **Best Season**: {best_season}")
                    st.write(f"• **Best Average**: {format_time(best_season_time)}")
                    
                    # Count runs by season
                    season_counts = best_periods['seasonal_data'].groupby('season')['run_count'].sum()
                    most_active_season = season_counts.idxmax()
                    st.write(f"• **Most Active**: {most_active_season} ({season_counts[most_active_season]} runs)")
    
    # Monthly heatmap
    st.subheader("🔥 Performance Heatmap")
    heatmap = create_monthly_heatmap(data)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)
        st.caption("🔴 Slower times → 🔵 Faster times. Darker blue = better performance!")
    
    # Improvement by season
    st.subheader("📈 Seasonal Improvement Trends")
    improvement_chart = create_improvement_by_season_chart(data)
    if improvement_chart:
        st.plotly_chart(improvement_chart, use_container_width=True)
        st.caption("Shows percentage improvement within each season (first run vs best run in that season)")
    
    # Detailed statistics tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Seasonal Statistics")
        if seasonal_chart and monthly_stats is not None:
            # Create seasonal summary
            seasonal_summary = monthly_stats.groupby([
                monthly_stats['month_name'].map({
                    'January': 'Winter', 'February': 'Winter', 'March': 'Spring',
                    'April': 'Spring', 'May': 'Spring', 'June': 'Summer',
                    'July': 'Summer', 'August': 'Summer', 'September': 'Autumn',
                    'October': 'Autumn', 'November': 'Autumn', 'December': 'Winter'
                }),
                'distance'
            ]).agg({
                'avg_time': 'mean',
                'best_time': 'min',
                'run_count': 'sum'
            }).round(2)
            
            seasonal_summary.index.names = ['Season', 'Distance']
            st.dataframe(seasonal_summary, use_container_width=True)
    
    with col2:
        st.subheader("🗓️ Monthly Breakdown")
        if monthly_stats is not None and len(monthly_stats) > 0:
            # Format monthly stats for display
            display_monthly = monthly_stats.copy()
            display_monthly['avg_time'] = display_monthly['avg_time'].apply(lambda x: f"{x:.1f}s")
            display_monthly['best_time'] = display_monthly['best_time'].apply(lambda x: f"{x:.1f}s")
            display_monthly.columns = ['Distance', 'Month', 'Avg Time', 'Best Time', 'Runs']
            
            st.dataframe(display_monthly, use_container_width=True)
    
    # Seasonal insights
    if best_periods:
        st.subheader("💡 Seasonal Insights")
        
        # Calculate some insights
        insights = []
        
        if 'seasonal_data' in best_periods:
            seasonal_data = best_periods['seasonal_data']
            
            # Find most consistent season (lowest std deviation)
            seasonal_consistency = seasonal_data.groupby('season')['avg_time'].std()
            most_consistent = seasonal_consistency.idxmin()
            insights.append(f"🎯 **Most Consistent Season**: {most_consistent} (lowest time variation)")
            
            # Find season with most improvement potential
            seasonal_ranges = seasonal_data.groupby('season')['avg_time'].apply(lambda x: x.max() - x.min())
            highest_range = seasonal_ranges.idxmax()
            if seasonal_ranges[highest_range] > 0:
                insights.append(f"📈 **Biggest Improvement Range**: {highest_range} ({seasonal_ranges[highest_range]:.1f}s spread)")
        
        # Display insights
        for insight in insights:
            st.markdown(insight)
    
    # Performance calendar view
    st.subheader("📅 Training Calendar Analysis")
    
    # Combine all data for calendar analysis
    all_data = []
    for distance, df in data.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['distance'] = distance
            all_data.append(df_copy)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        # Group by week of year
        combined['week'] = combined['date'].dt.isocalendar().week
        combined['year'] = combined['date'].dt.year
        
        weekly_stats = combined.groupby(['year', 'week']).agg({
            'time_seconds': ['count', 'mean'],
            'distance': lambda x: len(x.unique())
        }).round(2)
        
        weekly_stats.columns = ['Total Runs', 'Avg Time', 'Distances Trained']
        
        if len(weekly_stats) > 0:
            # Find most active weeks
            most_active_week = weekly_stats['Total Runs'].idxmax()
            best_performance_week = weekly_stats['Avg Time'].idxmin()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Most Active Week", 
                    f"Week {most_active_week[1]}, {most_active_week[0]}",
                    f"{weekly_stats.loc[most_active_week, 'Total Runs']} runs"
                )
            
            with col2:
                st.metric(
                    "Best Performance Week",
                    f"Week {best_performance_week[1]}, {best_performance_week[0]}",
                    f"{format_time(weekly_stats.loc[best_performance_week, 'Avg Time'])} avg"
                )
            
            with col3:
                def show_data_entry():
    """Show data entry page"""
    st.header("➕ Add New Running Data")
    
    # Information about data entry options
    st.info("""
    🎯 **Recommended Approach**: Add data directly to your Google Sheets for best results!
    
    **Why Google Sheets is better:**
    • ✅ Data persists permanently
    • ✅ Easy to edit/correct mistakes
    • ✅ Can add multiple entries quickly
    • ✅ Automatic backup
    • ✅ Access from any device
    """)
    
    # Quick link to Google Sheets
    st.markdown("""
    ### 🔗 Quick Access to Your Google Sheet
    [**Open Your Running Data Sheet →**](https://docs.google.com/spreadsheets/d/1CUM-P3wB2zxHrbmw1JM7vrxtqhtnsasy34NqaIkke_0/edit)
    
    **To add data in Google Sheets:**
    1. Click the link above
    2. Find an empty row
    3. Add your time and date in the correct columns
    4. Format: Time as `19.5` or `1.10.5`, Date as `DD-MM-YY`
    5. Return to this dashboard and refresh!
    """)
    
    st.divider()
    
    # Alternative: Quick entry form (demonstration purposes)
    st.subheader("🚀 Quick Entry Form")
    st.warning("""
    ⚠️ **Note**: This form is for demonstration only. 
    Data entered here will NOT be saved permanently and will disappear when you refresh the page.
    For permanent data storage, use Google Sheets above.
    """)
    
    with st.form("data_entry_form"):
        st.markdown("**Enter new running data:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            distance = st.selectbox(
                "Distance",
                ["100m", "200m", "300m", "400m", "500m"]
            )
            
            # Time input with format help
            time_input = st.text_input(
                "Time",
                placeholder="e.g., 19.5 or 1.10.5",
                help="Format: seconds (19.5) or minutes.seconds.tenths (1.10.5)"
            )
        
        with col2:
            date_input = st.date_input(
                "Date",
                value=datetime.now().date(),
                help="Date of the run"
            )
            
            notes = st.text_input(
                "Notes (optional)",
                placeholder="Weather, feeling, etc."
            )
        
        submitted = st.form_submit_button("Add Entry", type="primary")
        
        if submitted:
            if time_input and date_input:
                # Validate time format
                parsed_time = parse_time_to_seconds(time_input)
                if parsed_time is not None:
                    # Format date for display
                    formatted_date = date_input.strftime('%d-%m-%y')
                    
                    st.success(f"""
                    ✅ **Entry Added Successfully!**
                    
                    - **Distance**: {distance}
                    - **Time**: {format_time(parsed_time)} ({time_input})
                    - **Date**: {formatted_date}
                    - **Notes**: {notes if notes else "None"}
                    
                    ⚠️ **Remember**: This data is only temporary. 
                    Add it to your Google Sheet for permanent storage!
                    """)
                    
                    # Show how to add to Google Sheets
                    st.info(f"""
                    **To save permanently, add this to your Google Sheet:**
                    
                    1. Go to the {distance} columns in your sheet
                    2. In {distance}_Time column: `{time_input}`
                    3. In {distance}_Date column: `{formatted_date}`
                    """)
                else:
                    st.error("❌ Invalid time format. Use formats like: 19.5, 1.10.5, or 41.0")
            else:
                st.error("❌ Please fill in both time and date fields.")
    
    st.divider()
    
    # Data format guide
    st.subheader("📋 Data Format Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **⏱️ Time Formats:**
        - `19.5` = 19.5 seconds
        - `1.10.0` = 1 minute, 10.0 seconds  
        - `1.10.5` = 1 minute, 10.5 seconds
        - `41.0` = 41.0 seconds
        - `2.15.0` = 2 minutes, 15.0 seconds
        """)
    
    with col2:
        st.markdown("""
        **📅 Date Formats:**
        - Use: `DD-MM-YY` format
        - Example: `19-05-25` for May 19, 2025
        - Example: `03-12-24` for December 3, 2024
        """)
    
    # Show current data structure
    st.subheader("📊 Current Data Overview")
    
    # Load and display current data summary
    df = load_data()
    if df is not None:
        data = process_data(df)
        if data:
            summary_data = []
            for distance, distance_df in data.items():
                if not distance_df.empty:
                    summary_data.append({
                        'Distance': distance,
                        'Total Runs': len(distance_df),
                        'Latest Run': distance_df['date'].max().strftime('%d-%m-%Y'),
                        'Personal Record': format_time(distance_df['time_seconds'].min()),
                        'Last Time': format_time(distance_df.iloc[-1]['time_seconds'])
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("No data available yet. Add some runs to get started!")
    
    # Tips for efficient data entry
    st.subheader("💡 Pro Tips for Data Entry")
    
    tips = [
        "🏃‍♂️ **Add data immediately** after each run while it's fresh",
        "📱 **Use your phone** to add data to Google Sheets on the go",
        "🗓️ **Be consistent** with date formats (DD-MM-YY)",
        "⏱️ **Double-check times** for accuracy",
        "📝 **Add notes** about conditions (weather, how you felt, etc.)",
        "🔄 **Refresh the dashboard** after adding data to see updates",
        "💾 **Backup your data** occasionally by downloading the sheet"
    ]
    
    for tip in tips:
        st.markdown(tip)
    
    # Advanced data management
    with st.expander("🔧 Advanced Data Management"):
        st.markdown("""
        **Bulk Data Entry:**
        - Copy multiple rows at once in Google Sheets
        - Use autofill for dates (drag down from a cell)
        - Sort your data by date for better organization
        
        **Data Validation:**
        - Check for duplicate entries occasionally
        - Ensure all times are in consistent format
        - Verify dates are correct (easy to mix up day/month)
        
        **Export Options:**
        - Download your Google Sheet as Excel/CSV for backup
        - Share the sheet with family members
        - Use Google Sheets mobile app for quick entry
        
        **Common Mistakes to Avoid:**
        - Don't use different time formats in the same column
        - Don't leave empty rows between data
        - Don't change column headers (100m_Time, 100m_Date, etc.)
        """)

                    avg_runs_per_week = weekly_stats['Total Runs'].mean()
                st.metric(
                    "Avg Runs/Week",
                    f"{avg_runs_per_week:.1f}",
                    f"{len(weekly_stats)} weeks tracked"
                )

if __name__ == "__main__":
    main()
