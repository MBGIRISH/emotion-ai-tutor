"""
Adaptive LLM-powered tutoring system that responds to student emotions.
Generates personalized hints, explanations, and motivational responses.
"""

import os
from typing import Dict, Optional, List
from dotenv import load_dotenv
import openai
from backend.utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)


class AdaptiveTutor:
    """LLM-powered adaptive tutoring system"""
    
    def __init__(
        self,
        api_provider: str = "openai",  # "openai" or "anthropic"
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        """
        Initialize adaptive tutor.
        
        Args:
            api_provider: LLM API provider ("openai" or "anthropic")
            model: Model name to use
            temperature: Sampling temperature
        """
        self.api_provider = api_provider
        self.model = model
        self.temperature = temperature
        
        # Initialize API client
        if api_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
                self.client = openai.OpenAI(api_key=api_key)
            else:
                logger.warning("OPENAI_API_KEY not found. Tutor will use placeholder responses.")
                self.client = None
        elif api_provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
                else:
                    logger.warning("ANTHROPIC_API_KEY not found. Tutor will use placeholder responses.")
                    self.client = None
            except ImportError:
                logger.warning("anthropic package not installed. Install with: pip install anthropic")
                self.client = None
        else:
            self.client = None
            logger.warning(f"Unknown API provider: {api_provider}")
    
    def _get_emotion_context(self, emotions: Dict[str, float]) -> str:
        """
        Convert emotion dictionary to context string.
        
        Args:
            emotions: Emotion probabilities dictionary
            
        Returns:
            Formatted emotion context string
        """
        if not emotions:
            return "No emotion data available."
        
        # Get top emotions
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = sorted_emotions[:3]
        
        context = "Student's current emotional state: "
        context += ", ".join([f"{emotion} ({prob:.2f})" for emotion, prob in top_emotions])
        return context
    
    def _generate_adaptive_prompt(
        self,
        emotions: Dict[str, float],
        engagement: float,
        confusion: float,
        context: Optional[str] = None
    ) -> str:
        """
        Generate adaptive prompt based on student state.
        
        Args:
            emotions: Current emotion probabilities
            engagement: Engagement score (0-100)
            confusion: Confusion level (0-1)
            context: Optional additional context
            
        Returns:
            Formatted prompt for LLM
        """
        emotion_context = self._get_emotion_context(emotions)
        
        prompt = f"""You are an empathetic AI tutor helping a student learn. 

{emotion_context}
Engagement level: {engagement:.1f}/100
Confusion level: {confusion:.1f}

"""
        
        if confusion > 0.7:
            prompt += """The student appears highly confused. Please:
1. Break down the concept into simpler, smaller steps
2. Use analogies and examples
3. Ask if they need clarification
4. Be patient and encouraging
"""
        elif confusion > 0.4:
            prompt += """The student shows some confusion. Please:
1. Provide a simplified explanation
2. Offer a helpful hint
3. Check their understanding
"""
        
        if engagement < 30:
            prompt += """The student has low engagement. Please:
1. Use an encouraging, motivating tone
2. Try a different approach or example
3. Make the content more interactive or interesting
"""
        elif engagement > 70:
            prompt += """The student is highly engaged! Please:
1. Build on their enthusiasm
2. Provide more challenging content
3. Acknowledge their progress
"""
        
        if context:
            prompt += f"\nAdditional context: {context}\n"
        
        prompt += "\nGenerate a brief, helpful response (2-3 sentences) that adapts to the student's emotional state."
        
        return prompt
    
    def generate_response(
        self,
        emotions: Dict[str, float],
        engagement: float,
        confusion: float,
        context: Optional[str] = None
    ) -> str:
        """
        Generate adaptive tutor response based on student state.
        
        Args:
            emotions: Current emotion probabilities
            engagement: Engagement score (0-100)
            confusion: Confusion level (0-1)
            context: Optional additional context
            
        Returns:
            Tutor's adaptive response
        """
        if not self.client:
            # Return placeholder response if API not configured
            return self._get_placeholder_response(emotions, engagement, confusion)
        
        try:
            prompt = self._generate_adaptive_prompt(emotions, engagement, confusion, context)
            
            if self.api_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an empathetic, adaptive AI tutor."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
            
            elif self.api_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text.strip()
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._get_placeholder_response(emotions, engagement, confusion)
    
    def _get_placeholder_response(
        self,
        emotions: Dict[str, float],
        engagement: float,
        confusion: float
    ) -> str:
        """
        Generate placeholder response when LLM API is not available.
        
        Args:
            emotions: Current emotion probabilities
            engagement: Engagement score
            confusion: Confusion level
            
        Returns:
            Placeholder response
        """
        if confusion > 0.7:
            return "I notice you might be finding this challenging. Let me break this down into simpler steps. Would you like me to explain it differently?"
        elif confusion > 0.4:
            return "It looks like this concept might need a bit more clarification. Let me provide a simpler explanation with an example."
        elif engagement < 30:
            return "I see you might be feeling a bit disengaged. Let's try a different approach that might be more interesting. You've got this!"
        elif engagement > 70:
            top_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "engaged"
            if top_emotion in ["happy", "surprise"]:
                return "Great to see you're engaged! Let's build on this momentum and explore the next concept."
            else:
                return "You're doing well! Let's continue building on what you've learned."
        else:
            return "How are you feeling about this concept? Let me know if you need any clarification or want to move forward."
    
    def generate_chat_response(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate response to student's chat message with emotional context.
        
        Args:
            message: Student's message
            context: Optional context (emotions, engagement, confusion)
            
        Returns:
            Tutor's response
        """
        if not self.client:
            return "I'm here to help! Could you tell me more about what you're working on?"
        
        context_str = ""
        if context:
            emotions = context.get("emotions", {})
            engagement = context.get("engagement", 50.0)
            confusion = context.get("confusion", 0.0)
            context_str = self._get_emotion_context(emotions)
            context_str += f"\nEngagement: {engagement:.1f}/100, Confusion: {confusion:.1f}"
        
        try:
            prompt = f"""You are an empathetic AI tutor. A student sent you this message:

"{message}"

{context_str if context_str else ""}

Provide a helpful, adaptive response that considers the student's emotional state and engagement level."""
            
            if self.api_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an empathetic, adaptive AI tutor."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=300
                )
                return response.choices[0].message.content.strip()
            
            elif self.api_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text.strip()
        
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return "I'm here to help! Could you tell me more about what you're working on?"

