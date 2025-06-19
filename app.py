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
    """Create seasonal performance analysis"""
    all_data = []
    for distance, df in data.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['month'] = df_copy['date'].dt.month
            df_copy['season'] = df_copy['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            all_data.append(df_copy)
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    seasonal_avg = combined.groupby(['distance', 'season'])['time_seconds'].mean().reset_index()
    
    fig = px.bar(
        seasonal_avg,
        x='season',
        y='time_seconds',
        color='distance',
        title="Average Performance by Season",
        labels={'time_seconds': 'Average Time (seconds)'},
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
        ["📊 Overview", "📈 Individual Distances", "🌟 Performance Analysis", "📅 Seasonal Trends"]
    )
    
    if page == "📊 Overview":
        show_overview(data)
    elif page == "📈 Individual Distances":
        show_individual_distances(data)
    elif page == "🌟 Performance Analysis":
        show_performance_analysis(data)
    elif page == "📅 Seasonal Trends":
        show_seasonal_trends(data)

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
    """Show seasonal trends analysis"""
    st.header("📅 Seasonal Performance Trends")
    
    seasonal_chart = create_seasonal_analysis(data)
    if seasonal_chart:
        st.plotly_chart(seasonal_chart, use_container_width=True)
    
    # Monthly breakdown
    all_data = []
    for distance, df in data.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['month'] = df_copy['date'].dt.month
            df_copy['month_name'] = df_copy['date'].dt.strftime('%B')
            all_data.append(df_copy)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        # Monthly performance summary
        st.subheader("📊 Monthly Performance Summary")
        monthly_stats = combined.groupby(['distance', 'month_name']).agg({
            'time_seconds': ['count', 'mean', 'min']
        }).round(2)
        
        monthly_stats.columns = ['Runs', 'Avg Time (s)', 'Best Time (s)']
        st.dataframe(monthly_stats, use_container_width=True)
        
        # Best months analysis
        st.subheader("🏆 Best Performing Months")
        best_months = combined.groupby('month_name')['time_seconds'].mean().sort_values()
        
        fig = px.bar(
            x=best_months.index,
            y=best_months.values,
            title="Average Performance by Month (All Distances)",
            labels={'x': 'Month', 'y': 'Average Time (seconds)'}
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
