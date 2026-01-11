"""
Speech Emotion Recognition Model
CNN + BiLSTM + Attention architecture for 4-class emotion classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on important temporal features."""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_out):
        # lstm_out shape: (batch, seq_len, hidden_size * 2) for bidirectional
        # Compute attention scores
        attention_scores = self.attention_weights(lstm_out).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Apply attention weights
        attended = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_size * 2)
        
        return attended, attention_weights


class SpeechEmotionCNNBiLSTMAttention(nn.Module):
    """
    Deep learning model for Speech Emotion Recognition.
    Architecture: CNN layers -> BiLSTM -> Attention -> Dense layers
    """
    
    def __init__(self, input_dim: int = 181, num_classes: int = 4):
        """
        Args:
            input_dim: Input feature dimension (40 MFCCs + 12 Chroma + 128 Mel + 1 ZCR = 181)
            num_classes: Number of emotion classes (4)
        """
        super(SpeechEmotionCNNBiLSTMAttention, self).__init__()
        
        # Reshape input for CNN (treat features as 1D signal)
        # We'll use 1D convolutions on the feature vector
        self.input_dim = input_dim
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.dropout_conv = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool1d(100)  # Pool to fixed length
        
        # BiLSTM for temporal dependencies
        lstm_input_size = 256
        lstm_hidden_size = 128
        num_lstm_layers = 2
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(lstm_hidden_size * 2)  # *2 for bidirectional
        
        # Dense layers for classification
        dense_input_size = lstm_hidden_size * 2  # Output from attention
        self.fc1 = nn.Linear(dense_input_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch, input_dim)
        batch_size = x.size(0)
        
        # Reshape for CNN: (batch, 1, input_dim)
        x = x.unsqueeze(1)
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        
        # Pool to fixed length
        x = self.pool(x)  # (batch, 256, 100)
        
        # Transpose for LSTM: (batch, seq_len=100, features=256)
        x = x.transpose(1, 2)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, 100, hidden_size * 2)
        
        # Attention
        attended, attention_weights = self.attention(lstm_out)  # (batch, hidden_size * 2)
        
        # Dense layers
        x = F.relu(self.bn_fc1(self.fc1(attended)))
        x = self.dropout_fc1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        
        return x


if __name__ == "__main__":
    # Test model
    model = SpeechEmotionCNNBiLSTMAttention(input_dim=180, num_classes=4)
    
    # Test input
    test_input = torch.randn(4, 180)  # batch_size=4, feature_dim=180
    
    print("Model Architecture:")
    print(model)
    
    print(f"\nInput shape: {test_input.shape}")
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

