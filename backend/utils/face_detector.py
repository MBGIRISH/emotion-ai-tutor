"""
MediaPipe-based face detection and landmark extraction.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# MediaPipe is optional - use OpenCV if not available
try:
    import mediapipe as mp
    # Check if old API (solutions) is available
    if hasattr(mp, 'solutions'):
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
    else:
        # New MediaPipe API doesn't have solutions - use OpenCV fallback
        MEDIAPIPE_AVAILABLE = False
        mp_face_detection = None
        mp_face_mesh = None
        mp_drawing = None
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    mp_face_detection = None
    mp_face_mesh = None
    mp_drawing = None


class FaceDetector:
    """Face detection and landmark extraction using MediaPipe"""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0  # 0 for short-range, 1 for full-range
    ):
        """
        Initialize face detector (MediaPipe if available, else OpenCV).
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            model_selection: 0 for short-range, 1 for full-range
        """
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        
        if self.use_mediapipe:
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence
            )
            
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )
        else:
            # Use OpenCV Haar Cascade as fallback
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_detection = None
            self.face_mesh = None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image and return bounding boxes.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            List of bounding boxes (x, y, width, height)
        """
        if self.use_mediapipe:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            faces = []
            if results.detections:
                h, w = image.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    faces.append((x, y, width, height))
            
            return faces
        else:
            # Use OpenCV Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_detected = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return [(x, y, w, h) for (x, y, w, h) in faces_detected]
    
    def get_face_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face landmarks (468 points) from image.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Array of landmark coordinates (N, 2) or None if no face
        """
        if not self.use_mediapipe:
            # Return None if MediaPipe not available
            return None
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Extract landmark coordinates
            landmark_array = np.array([
                [lm.x * w, lm.y * h]
                for lm in landmarks.landmark
            ])
            return landmark_array
        
        return None
    
    def get_eye_landmarks(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get eye landmarks for blink detection.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Tuple of (left_eye_landmarks, right_eye_landmarks) or None
        """
        landmarks = self.get_face_landmarks(image)
        if landmarks is None:
            return None
        
        # MediaPipe face mesh eye indices
        # Left eye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Simplified: use 6 key points per eye
        left_eye_indices = [33, 7, 163, 144, 145, 153]
        right_eye_indices = [362, 382, 381, 380, 374, 373]
        
        left_eye = landmarks[left_eye_indices]
        right_eye = landmarks[right_eye_indices]
        
        return (left_eye, right_eye)
    
    def draw_face_detection(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw face bounding boxes on image.
        
        Args:
            image: BGR image
            faces: List of bounding boxes
            
        Returns:
            Image with drawn boxes
        """
        output = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return output
    
    def draw_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Draw face mesh landmarks on image.
        
        Args:
            image: BGR image
            
        Returns:
            Image with drawn landmarks
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        output = image.copy()
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    output,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
        return output

