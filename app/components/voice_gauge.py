"""
Voice emotion gauge component for Streamlit dashboard.
Displays real-time audio emotion probabilities.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict


class VoiceGauge:
    """Voice emotion gauge visualization component"""
    
    @staticmethod
    def display(emotions: Dict[str, float], frame_id: int = 0):
        """
        Display audio emotion probabilities as gauge chart.
        
        Args:
            emotions: Dictionary of audio emotion probabilities
        """
        # Sort emotions by probability
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Get top emotion
        top_emotion, top_prob = sorted_emotions[0]
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=top_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Voice: {top_emotion.capitalize()}"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        
        st.plotly_chart(fig, use_container_width=True, key=f"voice_gauge_{frame_id}")
        
        # Display all emotions as small bars (show all 4 emotions)
        labels = [emotion.capitalize() for emotion, _ in sorted_emotions]
        values = [prob * 100 for _, prob in sorted_emotions]
        
        fig2 = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color='steelblue',
                text=[f"{v:.1f}%" for v in values],
                textposition='outside'
            )
        ])
        
        fig2.update_layout(
            title="Voice Emotion Probabilities",
            xaxis_title="Emotion",
            yaxis_title="Probability (%)",
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True, key=f"voice_bar_{frame_id}")

