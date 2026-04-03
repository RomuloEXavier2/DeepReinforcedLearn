import torch
import torch.nn as nn


# ... (outros imports)

# NOVA ARQUITETURA DE REDE
class HybridDQN(nn.Module):
    def __init__(self, pallet_shape, vector_data_size, action_size, hidden_size=256):
        super(HybridDQN, self).__init__()

        # Ramo Convolucional (para o mapa do palete)
        self.pallet_channels, self.pallet_h, self.pallet_w = pallet_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(self.pallet_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Saída: 16x(H/2)x(W/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))  # Garante saída 64x2x2
        )
        cnn_out_size = 64 * 2 * 2

        # Ramo MLP (para dados vetoriais: caixas, etc.)
        self.mlp = nn.Sequential(
            nn.Linear(vector_data_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        mlp_out_size = 64

        # Camada final que combina as duas saídas
        self.final_head = nn.Sequential(
            nn.Linear(cnn_out_size + mlp_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, pallet_map_tensor, vector_data_tensor):
        cnn_features = self.cnn(pallet_map_tensor)
        cnn_features_flat = cnn_features.view(cnn_features.size(0), -1)

        mlp_features = self.mlp(vector_data_tensor)

        combined_features = torch.cat((cnn_features_flat, mlp_features), dim=1)

        return self.final_head(combined_features)

# O resto da classe do agente precisaria ser reescrito para lidar com
# a passagem dos dois tipos de estado separadamente para a rede.
# Isso demonstra a complexidade da mudança.