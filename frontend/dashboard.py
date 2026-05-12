import streamlit as st
import sqlite3
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
sys.path.append(r'D:\traffic_violation_system')

from database.models import DB_PATH
from database.db import get_all_challans, update_payment_status

# Page config
st.set_page_config(
    page_title="Traffic Violation Detection System",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Government Style CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f0f2f5;
    }

    /* Header */
    .gov-header {
        background: linear-gradient(135deg, #1a237e, #283593);
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        border-bottom: 4px solid #ff9800;
    }
    .gov-header h1 {
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin: 0;
        letter-spacing: 2px;
    }
    .gov-header p {
        color: #b0bec5;
        font-size: 14px;
        margin: 5px 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #1a237e;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #1a237e;
    }
    .metric-label {
        font-size: 13px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Alert box */
    .violation-alert {
        background: #fff3e0;
        border-left: 5px solid #ff5722;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .violation-paid {
        background: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }

    /* Section header */
    .section-header {
        background: #1a237e;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        margin: 15px 0 10px 0;
        letter-spacing: 1px;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #1a237e;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---- Plotly install check ----
try:
    import plotly.express as px
except:
    os.system("pip install plotly")
    import plotly.express as px

# ---- Database functions ----
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM challans")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM challans WHERE status='UNPAID'")
    unpaid = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM challans WHERE status='PAID'")
    paid = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(fine_amount) FROM challans")
    total_fine = cursor.fetchone()[0] or 0

    cursor.execute("SELECT SUM(fine_amount) FROM challans WHERE status='UNPAID'")
    pending_fine = cursor.fetchone()[0] or 0

    cursor.execute("SELECT SUM(fine_amount) FROM challans WHERE status='PAID'")
    collected_fine = cursor.fetchone()[0] or 0

    conn.close()
    return total, unpaid, paid, total_fine, pending_fine, collected_fine

def get_violation_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT violation_type, COUNT(*) FROM challans GROUP BY violation_type")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_daily_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT substr(timestamp, 1, 10) as date,
               COUNT(*) as count,
               SUM(fine_amount) as total_fine
        FROM challans
        GROUP BY date
        ORDER BY date
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def search_vehicle(vehicle_number):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM challans WHERE vehicle_number LIKE ?",
        (f'%{vehicle_number}%',)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("### 🚔 Navigation")
    page = st.radio("", [
        "📊 Dashboard",
        "🚨 All Violations",
        "🔍 Search Vehicle",
        "📈 Analytics"
    ])
    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    st.info("Camera: CAM-01\nLocation: Main Road\nStatus: 🟢 Active")
    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.rerun()

# ---- GOVERNMENT HEADER ----
st.markdown("""
<div class="gov-header">
    <h1>🚔 TRAFFIC POLICE DEPARTMENT</h1>
    <p>AI-Powered Traffic Violation Detection & E-Challan Management System</p>
    <p style="color: #ff9800; font-size: 12px;">
        REAL-TIME MONITORING | AUTOMATED CHALLAN GENERATION
    </p>
</div>
""", unsafe_allow_html=True)

# ---- GET DATA ----
total, unpaid, paid, total_fine, pending_fine, collected_fine = get_stats()

# ---- PAGE: DASHBOARD ----
if page == "📊 Dashboard":

    # Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Challans</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f44336;">
            <div class="metric-value" style="color: #f44336;">{unpaid}</div>
            <div class="metric-label">Unpaid</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #4caf50;">
            <div class="metric-value" style="color: #4caf50;">{paid}</div>
            <div class="metric-label">Paid</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ff9800;">
            <div class="metric-value" style="color: #ff9800;">₹{total_fine:,}</div>
            <div class="metric-label">Total Fine</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f44336;">
            <div class="metric-value" style="color: #f44336;">₹{pending_fine:,}</div>
            <div class="metric-label">Pending</div>
        </div>""", unsafe_allow_html=True)

    with c6:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #4caf50;">
            <div class="metric-value" style="color: #4caf50;">₹{collected_fine:,}</div>
            <div class="metric-label">Collected</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Violations by Type</div>',
                   unsafe_allow_html=True)
        violation_data = get_violation_data()
        if violation_data:
            df_v = pd.DataFrame(violation_data, columns=['Violation', 'Count'])
            fig = px.pie(df_v, values='Count', names='Violation',
                        color_discrete_sequence=px.colors.sequential.Blues_r,
                        hole=0.4)
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Koi data nahi abhi tak!")

    with col2:
        st.markdown('<div class="section-header">💰 Payment Status</div>',
                   unsafe_allow_html=True)
        if total > 0:
            fig2 = go.Figure(data=[go.Bar(
                x=['Paid', 'Unpaid'],
                y=[paid, unpaid],
                marker_color=['#4caf50', '#f44336'],
                text=[f'₹{collected_fine:,}', f'₹{pending_fine:,}'],
                textposition='auto'
            )])
            fig2.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis_title="Count"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Koi data nahi abhi tak!")

    # Recent Challans
    st.markdown('<div class="section-header">🚨 Recent Violations</div>',
               unsafe_allow_html=True)

    challans = get_all_challans()
    if challans:
        for c in challans[:5]:  # Sirf last 5
            status_class = "violation-paid" if c[8] == 'PAID' else "violation-alert"
            st.markdown(f"""
            <div class="{status_class}">
                🚗 <b>{c[2]}</b> &nbsp;|&nbsp;
                ⚠️ {c[3]} &nbsp;|&nbsp;
                💰 ₹{c[4]} &nbsp;|&nbsp;
                📅 {c[6]} &nbsp;|&nbsp;
                {'✅ PAID' if c[8]=='PAID' else '❌ UNPAID'}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Koi challan nahi mila!")

# ---- PAGE: ALL VIOLATIONS ----
elif page == "🚨 All Violations":
    st.markdown('<div class="section-header">🚨 All Violation Records</div>',
               unsafe_allow_html=True)

    challans = get_all_challans()

    if not challans:
        st.info("Koi challan nahi mila!")
    else:
        for c in challans:
            status_color = "🟢" if c[8] == 'PAID' else "🔴"
            with st.expander(
                f"{status_color} {c[2]} — {c[3]} — ₹{c[4]} — {c[8]}"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Challan ID:** `{c[1]}`")
                    st.markdown(f"**Vehicle:** {c[2]}")
                    st.markdown(f"**Violation:** {c[3]}")
                with col2:
                    st.markdown(f"**Fine:** ₹{c[4]}")
                    st.markdown(f"**Location:** {c[5]}")
                    st.markdown(f"**Time:** {c[6]}")
                with col3:
                    st.markdown(f"**Status:** {c[8]}")

                    # PDF Download
                    if os.path.exists(c[7]):
                        with open(c[7], 'rb') as f:
                            st.download_button(
                                label="📄 Download PDF",
                                data=f,
                                file_name=f"challan_{c[1]}.pdf",
                                mime="application/pdf",
                                key=f"dl_{c[1]}"
                            )

                    # Payment button
                    if c[8] == 'UNPAID':
                        if st.button("✅ Mark PAID",
                                    key=f"pay_{c[1]}"):
                            update_payment_status(c[1], 'PAID')
                            st.success("Payment updated!")
                            st.rerun()

# ---- PAGE: SEARCH ----
elif page == "🔍 Search Vehicle":
    st.markdown('<div class="section-header">🔍 Search Vehicle</div>',
               unsafe_allow_html=True)

    search = st.text_input("Vehicle Number Enter Karo",
                          placeholder="e.g. UP80AB1234")

    if search:
        results = search_vehicle(search)
        if results:
            st.success(f"{len(results)} challan(s) mile!")
            for c in results:
                status_color = "🟢" if c[8] == 'PAID' else "🔴"
                with st.expander(
                    f"{status_color} {c[2]} — ₹{c[4]} — {c[8]}"
                ):
                    st.markdown(f"**Challan ID:** `{c[1]}`")
                    st.markdown(f"**Violation:** {c[3]}")
                    st.markdown(f"**Fine:** ₹{c[4]}")
                    st.markdown(f"**Time:** {c[6]}")
                    st.markdown(f"**Status:** {c[8]}")

                    if os.path.exists(c[7]):
                        with open(c[7], 'rb') as f:
                            st.download_button(
                                label="📄 Download PDF",
                                data=f,
                                file_name=f"challan_{c[1]}.pdf",
                                mime="application/pdf",
                                key=f"s_{c[1]}"
                            )
        else:
            st.warning("Koi challan nahi mila is vehicle ka!")

# ---- PAGE: ANALYTICS ----
elif page == "📈 Analytics":
    st.markdown('<div class="section-header">📈 Analytics & Reports</div>',
               unsafe_allow_html=True)

    daily_data = get_daily_data()

    if daily_data:
        df_d = pd.DataFrame(daily_data,
                           columns=['Date', 'Challans', 'Fine Collected'])

        # Daily challans chart
        st.markdown("**Daily Challans**")
        fig3 = px.bar(df_d, x='Date', y='Challans',
                     color_discrete_sequence=['#1a237e'])
        fig3.update_layout(paper_bgcolor='white', plot_bgcolor='white')
        st.plotly_chart(fig3, use_container_width=True)

        # Daily fine chart
        st.markdown("**Daily Fine Collected**")
        fig4 = px.line(df_d, x='Date', y='Fine Collected',
                      markers=True,
                      color_discrete_sequence=['#ff9800'])
        fig4.update_layout(paper_bgcolor='white', plot_bgcolor='white')
        st.plotly_chart(fig4, use_container_width=True)

        # Summary table
        st.markdown("**Summary Table**")
        st.dataframe(df_d, use_container_width=True)
    else:
        st.info("Analytics ke liye pehle kuch challans generate karo!")

    # Fine rules table
    st.markdown('<div class="section-header">📋 Fine Rules</div>',
               unsafe_allow_html=True)
    fine_data = {
        'Violation': ['No Helmet', 'No Seatbelt', 'Red Light',
                     'Over Speeding', 'Wrong Way', 'No Parking'],
        'Fine (₹)': [500, 500, 1000, 1500, 2000, 300],
        'Section': ['MV Act 129', 'MV Act 138',
                   'MV Act 119', 'MV Act 112',
                   'MV Act 184', 'MV Act 122']
    }
    df_fine = pd.DataFrame(fine_data)
    st.dataframe(df_fine, use_container_width=True, hide_index=True)