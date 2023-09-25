import torch

class LSTM(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        
        # Create vocab embedding
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, embedding_dim, batch_first=True, num_layers=2, bidirectional=True, dropout=0.1)

        # Linear readouts (arbitrary dims for now)
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 151)

        # Activations
        self.relu = torch.nn.ReLU()

        # Training stabilization
        self.bn = torch.nn.BatchNorm1d(hidden_dim)

        # Countering overfitting
        self.dropout = torch.nn.Dropout(0.12)
        
    def forward(self, vectorized_text):
        x = self.embeddings(vectorized_text)
        self.dropout(x)

        # parse out the multi-dimensional hidden output
        output, (hidden, cell) = self.lstm(x)

        # reduce the old-dimensionality!
        hidden = hidden[0].squeeze(0)
        
        hidden = self.bn(self.relu(self.linear(hidden)))
        return self.linear2(hidden)



