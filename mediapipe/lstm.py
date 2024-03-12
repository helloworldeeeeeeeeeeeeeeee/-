import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get output at the last time step
        out = out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)
        return out


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Create a bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # Linear layer for classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 because of bidirectional

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get output at the last time step
        out = out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)
        return out

class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Create a bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # Linear layer for classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 because of bidirectional

        # Attention layers
        self.attention_w = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.attention_u = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.attention_v = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Compute attention scores
        attention_weights = torch.tanh(self.attention_w(out)) * torch.sigmoid(self.attention_u(out))
        attention_scores = self.attention_v(attention_weights).squeeze(2)

        # Apply softmax to get attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=1)

        # Apply attention to LSTM outputs
        attention_out = torch.bmm(attention_probs.unsqueeze(1), out).squeeze(1)

        # Pass attention output through fully connected layer
        out = self.fc(attention_out)
        return out

    '''
    在这个示例代码中，我们添加了三个额外的线性层来计算交叉注意力机制的注意力分数，然后将注意力应用于LSTM的输出。

    '''


class BiLSTMAttention_TTA(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMAttention_TTA, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Create a bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # Linear layer for classification
        self.fc = nn.Linear(hidden_size * 4, num_classes)  # Multiply by 4 because of bidirectional and concatenation

        # Attention layers
        self.attention_w = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.attention_u = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.attention_v = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Reverse the input sequence
        x_reverse = torch.flip(out, [1])

        # Compute attention scores for original and reversed sequences
        attention_weights = torch.tanh(self.attention_w(out)) * torch.sigmoid(self.attention_u(out))
        attention_scores = self.attention_v(attention_weights).squeeze(2)
        attention_probs = torch.softmax(attention_scores, dim=1)

        attention_weights_reverse = torch.tanh(self.attention_w(x_reverse)) * torch.sigmoid(self.attention_u(x_reverse))
        attention_scores_reverse = self.attention_v(attention_weights_reverse).squeeze(2)
        attention_probs_reverse = torch.softmax(attention_scores_reverse, dim=1)

        # Apply attention to LSTM outputs
        attention_out = torch.bmm(attention_probs.unsqueeze(1), out).squeeze(1)
        attention_out_reverse = torch.bmm(attention_probs_reverse.unsqueeze(1), x_reverse).squeeze(1)

        # Concatenate original and reversed attention outputs
        combined_out = torch.cat((attention_out, attention_out_reverse), dim=1)

        # Pass combined output through fully connected layer
        out = self.fc(combined_out)
        return out

class BiLSTMAttention_TTA_plus(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMAttention_TTA_plus, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Create a bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # Linear layer for classification
        self.fc = nn.Linear(hidden_size * 4, num_classes)  # Multiply by 2 because of bidirectional

        # Attention layers
        self.attention_w = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.attention_u = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.attention_v = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Generate reverse sequence
        x_reverse = torch.flip(x, [1])

        # Initialize hidden and cell states for forward and reverse LSTMs
        h0_forward = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        c0_forward = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        h0_reverse = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        c0_reverse = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional

        # Forward pass through forward and reverse LSTMs
        out_forward, _ = self.lstm(x, (h0_forward, c0_forward))
        out_reverse, _ = self.lstm(x_reverse, (h0_reverse, c0_reverse))

        # Reverse the reverse sequence
        out_reverse = torch.flip(out_reverse, [1])

        # Compute attention scores for both forward and reverse outputs
        attention_weights_forward = torch.tanh(self.attention_w(out_forward)) * torch.sigmoid(self.attention_u(out_forward))
        attention_scores_forward = self.attention_v(attention_weights_forward).squeeze(2)
        attention_probs_forward = torch.softmax(attention_scores_forward, dim=1)

        attention_weights_reverse = torch.tanh(self.attention_w(out_reverse)) * torch.sigmoid(self.attention_u(out_reverse))
        attention_scores_reverse = self.attention_v(attention_weights_reverse).squeeze(2)
        attention_probs_reverse = torch.softmax(attention_scores_reverse, dim=1)

        # Apply attention to LSTM outputs
        attention_out_forward = torch.bmm(attention_probs_forward.unsqueeze(1), out_forward).squeeze(1)
        attention_out_reverse = torch.bmm(attention_probs_reverse.unsqueeze(1), out_reverse).squeeze(1)

        # Concatenate forward and reverse attention outputs
        combined_out = torch.cat((attention_out_forward, attention_out_reverse), dim=1)

        # Pass combined output through fully connected layer
        out = self.fc(combined_out)
        return out

class EnhancedClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EnhancedClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # BiLSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # Convolutional layer
        self.conv = nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # Convolutional layer
        conv_out = self.conv(lstm_out.permute(0, 2, 1))

        # Max pooling
        pool_out = self.pool(conv_out)

        # Flatten
        flat_out = pool_out.view(pool_out.size(0), -1)

        # Fully connected layers with dropout
        fc1_out = self.fc1(flat_out)
        fc1_out = self.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)

        return fc2_out
class DoubleBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DoubleBiLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # First BiLSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # Second BiLSTM layer
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 because of bidirectional

    def forward(self, x):
        # First BiLSTM layer
        lstm1_out, _ = self.lstm1(x)

        # Second BiLSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)

        # Get output at the last time step
        final_out = lstm2_out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(final_out)
        return out

class MultiLayerBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MultiLayerBiLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Create a list to hold the layers
        self.lstms = nn.ModuleList()

        # Add the first layer
        self.lstms.append(nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True))

        # Add additional layers
        for _ in range(num_layers - 1):
            self.lstms.append(nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True))

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 because of bidirectional

    def forward(self, x):
        # Iterate through the LSTM layers
        for lstm in self.lstms:
            lstm_out, _ = lstm(x)
            x = lstm_out

        # Get output at the last time step
        final_out = x[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(final_out)
        return out


if __name__ == '__main__':

    # Hyperparameters
    input_size = 150
    hidden_size = 64
    num_classes = 226  # Replace with the actual number of classes
    batch_size = 32

    # Create the LSTM model
    model = LSTMClassifier(input_size, hidden_size, num_classes)
    print(model)

    # Dummy input data
    input_data = torch.randn(batch_size, 18, input_size)

    # Forward pass
    output = model(input_data)
    print("Output shape:", output.shape)

    # 创建一个 MultiLayerBiLSTMClassifier 实例
    input_size = 10  # 输入特征的维度
    hidden_size = 64  # 隐藏层大小
    num_layers = 3  # LSTM 层数
    num_classes = 2  # 输出类别数量
    model = MultiLayerBiLSTMClassifier(input_size, hidden_size, num_layers, num_classes)
