import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="AI Student Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

@st.cache_data
def load_dataset():
    return pd.read_csv('student_data.csv')

def get_risk_color(risk_label):
    colors = {'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'}
    return colors.get(risk_label, '#6c757d')

def get_risk_emoji(risk_label):
    emojis = {'Low Risk': '🟢', 'Medium Risk': '🟡', 'High Risk': '🔴'}
    return emojis.get(risk_label, '⚪')

def generate_recommendations(input_data):
    recommendations = []
    risk_factors = []
    
    if input_data['Attendance'] < 75:
        risk_factors.append("Low attendance")
        recommendations.append({
            'area': 'Attendance',
            'current': f"{input_data['Attendance']:.1f}%",
            'target': '≥ 75%',
            'action': 'Attend classes regularly. Set reminders and prioritize core subjects.',
            'priority': 'High'
        })
    
    if input_data['Internal_Marks'] < 65:
        risk_factors.append("Low internal marks")
        recommendations.append({
            'area': 'Internal Marks',
            'current': f"{input_data['Internal_Marks']:.1f}%",
            'target': '≥ 65%',
            'action': 'Review class notes, practice past papers, attend remedial classes.',
            'priority': 'High'
        })
    
    if input_data['Previous_CGPA'] < 6.5:
        risk_factors.append("Low previous CGPA")
        recommendations.append({
            'area': 'Academic Performance',
            'current': f"{input_data['Previous_CGPA']:.2f}",
            'target': '≥ 6.5',
            'action': 'Focus on improving grades in current semester. Seek tutoring for difficult subjects.',
            'priority': 'Medium'
        })
    
    if input_data['Assignment_Score'] < 70:
        risk_factors.append("Low assignment scores")
        recommendations.append({
            'area': 'Assignments',
            'current': f"{input_data['Assignment_Score']:.1f}%",
            'target': '≥ 70%',
            'action': 'Start assignments early, clarify doubts with faculty, use library resources.',
            'priority': 'High'
        })
    
    if input_data['Assignment_Submission'] < 85:
        risk_factors.append("Incomplete assignment submissions")
        recommendations.append({
            'area': 'Assignment Submission',
            'current': f"{input_data['Assignment_Submission']:.1f}%",
            'target': '≥ 85%',
            'action': 'Create a submission calendar, set early deadlines, avoid last-minute rush.',
            'priority': 'High'
        })
    
    if input_data['Study_Hours'] < 2.5:
        risk_factors.append("Insufficient study time")
        recommendations.append({
            'area': 'Study Hours',
            'current': f"{input_data['Study_Hours']:.1f} hrs/day",
            'target': '≥ 2.5 hrs/day',
            'action': 'Create a daily study schedule, use Pomodoro technique, eliminate distractions.',
            'priority': 'High'
        })
    
    if input_data['Backlogs'] > 0:
        risk_factors.append(f"Existing backlogs ({input_data['Backlogs']})")
        recommendations.append({
            'area': 'Backlogs',
            'current': f"{int(input_data['Backlogs'])} backlog(s)",
            'target': '0 backlogs',
            'action': 'Meet academic advisor for backlog clearance plan. Allocate dedicated weekly time.',
            'priority': 'Critical'
        })
    
    if input_data['Sleep_Hours'] < 6:
        risk_factors.append("Insufficient sleep")
        recommendations.append({
            'area': 'Sleep',
            'current': f"{input_data['Sleep_Hours']:.1f} hrs/night",
            'target': '≥ 7 hrs/night',
            'action': 'Maintain consistent sleep schedule, avoid caffeine late afternoon, limit screen time before bed.',
            'priority': 'Medium'
        })
    
    if input_data['Previous_Failures'] > 0:
        risk_factors.append(f"Previous failures ({int(input_data['Previous_Failures'])})")
        recommendations.append({
            'area': 'Past Failures',
            'current': f"{int(input_data['Previous_Failures'])} failed subject(s)",
            'target': 'No failures',
            'action': 'Identify root causes, retake failed subjects, seek mentor guidance.',
            'priority': 'Medium'
        })
    
    if not risk_factors:
        risk_factors.append("No significant risk factors detected")
        recommendations.append({
            'area': 'Maintenance',
            'current': 'Good standing',
            'target': 'Maintain performance',
            'action': 'Continue current study habits. Stay consistent with attendance and assignments.',
            'priority': 'Low'
        })
    
    return risk_factors, recommendations

def create_probability_chart(probabilities):
    fig = go.Figure(data=[
        go.Bar(
            x=['Low Risk', 'Medium Risk', 'High Risk'],
            y=[probabilities[0]*100, probabilities[1]*100, probabilities[2]*100],
            marker_color=['#28a745', '#ffc107', '#dc3545'],
            text=[f'{probabilities[0]*100:.1f}%', f'{probabilities[1]*100:.1f}%', f'{probabilities[2]*100:.1f}%'],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="Risk Probability Distribution",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=350,
        showlegend=False
    )
    return fig

def create_radar_chart(input_data, features):
    student_values = [input_data[f] for f in features]
    
    max_vals = {
        'Attendance': 100, 'Internal_Marks': 100, 'Previous_CGPA': 10,
        'Assignment_Score': 100, 'Assignment_Submission': 100,
        'Study_Hours': 10, 'Backlogs': 5, 'Sleep_Hours': 12, 'Previous_Failures': 4
    }
    
    normalized = [(student_values[i] / max_vals[features[i]]) * 100 for i in range(len(features))]
    normalized[-2] = 100 - normalized[-2]  # Backlogs - invert
    normalized[-1] = 100 - normalized[-1]  # Previous_Failures - invert
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized,
        theta=features,
        fill='toself',
        name='Student Profile',
        line_color='#1f77b4'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=400,
        title="Student Profile Radar Chart"
    )
    return fig

def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background: #fff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .priority-high { border-left: 4px solid #dc3545; }
    .priority-medium { border-left: 4px solid #ffc107; }
    .priority-low { border-left: 4px solid #28a745; }
    .priority-critical { border-left: 4px solid #721c24; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🎓 AI Student Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Early-Warning & Personalized Academic Support System</p>', unsafe_allow_html=True)
    
    model, scaler, features = load_model_and_scaler()
    df = load_dataset()
    
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("📝 Student Information")
            
            with st.form("prediction_form"):
                student_id = st.text_input("Student ID", value="STU1001")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    attendance = st.slider("Attendance (%)", 0, 100, 75)
                    internal_marks = st.slider("Internal Marks (%)", 0, 100, 65)
                    prev_cgpa = st.slider("Previous CGPA", 0.0, 10.0, 7.0, 0.1)
                    assignment_score = st.slider("Assignment Score (%)", 0, 100, 70)
                with col_b:
                    assignment_submission = st.slider("Assignment Submission (%)", 0, 100, 80)
                    study_hours = st.slider("Study Hours/Day", 0.0, 10.0, 3.0, 0.5)
                    backlogs = st.number_input("Backlogs", 0, 10, 0)
                    sleep_hours = st.slider("Sleep Hours/Night", 0.0, 12.0, 7.0, 0.5)
                
                prev_failures = st.number_input("Previous Failures", 0, 10, 0)
                
                submitted = st.form_submit_button("🔮 Predict Risk", type="primary", use_container_width=True)
        
        with col2:
            if submitted:
                input_data = {
                    'Attendance': attendance,
                    'Internal_Marks': internal_marks,
                    'Previous_CGPA': prev_cgpa,
                    'Assignment_Score': assignment_score,
                    'Assignment_Submission': assignment_submission,
                    'Study_Hours': study_hours,
                    'Backlogs': backlogs,
                    'Sleep_Hours': sleep_hours,
                    'Previous_Failures': prev_failures
                }
                
                input_df = pd.DataFrame([input_data], columns=features)
                input_scaled = scaler.transform(input_df)
                
                prediction = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                
                risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
                risk_label = risk_labels[prediction]
                risk_color = get_risk_color(risk_label)
                risk_emoji = get_risk_emoji(risk_label)
                
                st.markdown(f"""
                <div class="risk-box" style="background: {risk_color}20; border: 2px solid {risk_color};">
                    <h1>{risk_emoji} {risk_label}</h1>
                    <h3>Student: {student_id}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(create_probability_chart(probabilities), use_container_width=True)
                
                st.subheader("🎯 Key Risk Factors")
                risk_factors, recommendations = generate_recommendations(input_data)
                
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
                
                st.subheader("💡 Personalized Recommendations")
                
                priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
                recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
                
                for rec in recommendations:
                    priority_class = f"priority-{rec['priority'].lower()}"
                    st.markdown(f"""
                    <div class="recommendation-card {priority_class}">
                        <h4>{rec['area']} <span style="float: right; color: #666;">Priority: {rec['priority']}</span></h4>
                        <p><strong>Current:</strong> {rec['current']} &nbsp;|&nbsp; <strong>Target:</strong> {rec['target']}</p>
                        <p>{rec['action']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📈 Dataset Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Low Risk", len(df[df['Risk_Label']=='Low Risk']))
        with col3:
            st.metric("Medium Risk", len(df[df['Risk_Label']=='Medium Risk']))
        with col4:
            st.metric("High Risk", len(df[df['Risk_Label']=='High Risk']))
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df, names='Risk_Label', title='Risk Distribution',
                         color='Risk_Label',
                         color_discrete_map={'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='Attendance', color='Risk_Label', nbins=20,
                               title='Attendance Distribution by Risk',
                               color_discrete_map={'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='Risk_Label', y='Internal_Marks', color='Risk_Label',
                         title='Internal Marks by Risk Category',
                         color_discrete_map={'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='Risk_Label', y='Study_Hours', color='Risk_Label',
                         title='Study Hours by Risk Category',
                         color_discrete_map={'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔗 Feature Correlations")
        corr_features = ['Attendance', 'Internal_Marks', 'Previous_CGPA', 'Assignment_Score',
                         'Assignment_Submission', 'Study_Hours', 'Backlogs', 'Sleep_Hours', 'Previous_Failures']
        corr_matrix = df[corr_features].corr()
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("ℹ️ About This System")
        st.markdown("""
        ### AI Student Risk Predictor
        
        An intelligent early-warning and personalized academic support system that uses Machine Learning 
        to identify students at academic risk and provide actionable recommendations.
        
        **How it works:**
        1. **Input** student academic and behavioral data
        2. **ML Model** (Random Forest) processes the features
        3. **Prediction** classifies risk as Low, Medium, or High
        4. **Probability** scores show confidence levels
        5. **Risk Factors** identify specific problem areas
        6. **Recommendations** provide personalized action plans
        
        **Features Analyzed:**
        - Attendance percentage
        - Internal examination marks
        - Previous CGPA
        - Assignment performance & submission rate
        - Daily study hours
        - Current backlogs
        - Sleep patterns
        - Previous subject failures
        
        **Technology Stack:**
        - Python, Pandas, NumPy
        - Scikit-learn (Random Forest)
        - Streamlit (Dashboard)
        - Plotly (Visualizations)
        - Joblib (Model persistence)
        
        **Model Performance:**
        - Trained on 2,000 synthetic student records
        - 80/20 train-test split
        - Evaluated with Accuracy, Precision, Recall, F1-Score
        """)

if __name__ == "__main__":
    main()