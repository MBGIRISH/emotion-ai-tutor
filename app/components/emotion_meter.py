"""
Emotion meter component for Streamlit dashboard.
Displays real-time face emotion probabilities as bar chart.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict


class EmotionMeter:
    """Emotion meter visualization component"""
    
    @staticmethod
    def display(emotions: Dict[str, float], frame_id: int = 0):
        """
        Display emotion probabilities as horizontal bar chart.
        
        Args:
            emotions: Dictionary of emotion probabilities
        """
        # Sort emotions by probability
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Extract labels and values
        labels = [emotion.capitalize() for emotion, _ in sorted_emotions]
        values = [prob for _, prob in sorted_emotions]
        
        # Color mapping for emotions
        color_map = {
            "Happy": "#FFD700",
            "Sad": "#4169E1",
            "Angry": "#FF4500",
            "Fear": "#8B0000",
            "Surprise": "#FF69B4",
            "Disgust": "#228B22",
            "Neutral": "#808080"
        }
        
        colors = [color_map.get(label, "#808080") for label in labels]
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{v:.2f}" for v in values],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Probability: %{x:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Face Emotion Probabilities",
            xaxis_title="Probability",
            yaxis_title="Emotion",
            height=300,
            xaxis=dict(range=[0, 1]),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"emotion_meter_{frame_id}")
        
        # Display top emotion
        top_emotion, top_prob = sorted_emotions[0]
        st.metric(
            label="Dominant Emotion",
            value=f"{top_emotion.capitalize()} ({top_prob:.2%})"
        )

