"""
Engagement bar component for Streamlit dashboard.
Displays engagement score and confusion level.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Optional


class EngagementBar:
    """Engagement bar visualization component"""
    
    @staticmethod
    def display(engagement_score: float, confusion_level: float, frame_id: int = 0):
        """
        Display engagement score and confusion level.
        
        Args:
            engagement_score: Engagement score (0-100)
            confusion_level: Confusion level (0-1)
        """
        # Engagement score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=engagement_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Engagement Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': EngagementBar._get_engagement_color(engagement_score)},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 60], 'color': "gray"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        
        st.plotly_chart(fig, use_container_width=True, key=f"eng_gauge_{frame_id}")
        
        # Confusion level bar
        confusion_percent = confusion_level * 100
        
        fig2 = go.Figure(data=[
            go.Bar(
                x=["Confusion Level"],
                y=[confusion_percent],
                marker_color=EngagementBar._get_confusion_color(confusion_level),
                text=[f"{confusion_percent:.1f}%"],
                textposition='outside',
                width=[0.5]
            )
        ])
        
        fig2.update_layout(
            title="Confusion Level",
            yaxis_title="Confusion (%)",
            yaxis=dict(range=[0, 100]),
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True, key=f"conf_bar_{frame_id}")
        
        # Status indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Engagement", f"{engagement_score:.1f}/100")
        
        with col2:
            st.metric("Confusion", f"{confusion_percent:.1f}%")
        
        # Status message
        if confusion_level > 0.7:
            st.error("⚠️ High confusion detected! Consider simplifying the explanation.")
        elif confusion_level > 0.4:
            st.warning("⚠️ Moderate confusion detected. Student may need additional support.")
        elif engagement_score < 30:
            st.info("💡 Low engagement. Try a different approach to maintain interest.")
        elif engagement_score > 70:
            st.success("✅ High engagement! Student is actively learning.")
    
    @staticmethod
    def _get_engagement_color(score: float) -> str:
        """Get color based on engagement score"""
        if score < 30:
            return "red"
        elif score < 60:
            return "orange"
        else:
            return "green"
    
    @staticmethod
    def _get_confusion_color(level: float) -> str:
        """Get color based on confusion level"""
        if level < 0.3:
            return "green"
        elif level < 0.7:
            return "orange"
        else:
            return "red"

