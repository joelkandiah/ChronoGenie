import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_layers, input_dim, num_outputs, 
                 prediction_distribution_types=None, scalers=None):
        super().__init__()
        self.num_outputs = num_outputs
        self.prediction_distribution_types = prediction_distribution_types
        
        layers = []
        prev_dim = input_dim
        
        for layer_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.2))
            prev_dim = layer_dim
        self.parameter_net = nn.Sequential(*layers)          

        # Creating two heads: one for mu and one for dispersion
        self.mu_head = nn.Linear(prev_dim, num_outputs)
        self.disp_head = nn.Linear(prev_dim, num_outputs)

        # Extract and store scaling parameters as buffers if provided
        if scalers is not None:
            means = []
            stds = []
            for i in range(num_outputs):
                if i < len(scalers) and scalers[i] is not None and hasattr(scalers[i], 'mean_'):
                    means.append(scalers[i].mean_[0])
                    stds.append(scalers[i].scale_[0])
                else:
                    means.append(0.0)
                    stds.append(1.0)
            
            self.register_buffer("means", torch.tensor(means, dtype=torch.float32))
            self.register_buffer("stds", torch.tensor(stds, dtype=torch.float32))

    def scale(self, x_raw, indices=None):
        """Scales raw data using stored means/stds buffers."""
        if not hasattr(self, "means"):
            return x_raw
        
        means = self.means if indices is None else self.means[indices]
        stds = self.stds if indices is None else self.stds[indices]
        
        shape_view = [1] * (x_raw.ndim - 1) + [-1]
        x_scaled = (x_raw - means.view(*shape_view)) / stds.view(*shape_view)
        return x_scaled

    def unscale(self, x_scaled, indices=None):
        """Unscales data using stored means/stds buffers."""
        if not hasattr(self, "means"):
            return x_scaled
            
        means = self.means if indices is None else self.means[indices]
        stds = self.stds if indices is None else self.stds[indices]
        
        shape_view = [1] * (x_scaled.ndim - 1) + [-1]
        x_unscaled = x_scaled * stds.view(*shape_view) + means.view(*shape_view)
        return x_unscaled
        
    def forward(self, data):
        shared = self.parameter_net(data) # [batch_size, num_nodes, hidden_dim]
        mu = self.mu_head(shared)
        var = self.disp_head(shared)

        return mu, var

class GNNEncoder(nn.Module):
    def __init__(self, num_features, layer_specs, dropout=0.1, self_loop:bool=True, use_edge_attr:bool=False, edge_attr_dim=None, residual_connection:bool=False):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.edge_attr_dim = edge_attr_dim if use_edge_attr else None

        layers = []
        prev_dim = num_features
        
        for i, layer_info in enumerate(layer_specs):
            out_dim = layer_info[0]
            num_heads = layer_info[1]
            concat = layer_info[2]

            if self.use_edge_attr and (self.edge_attr_dim is not None):
                conv = GATv2Conv(
                    in_channels=prev_dim, 
                    out_channels=out_dim,
                    heads=num_heads, 
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=self_loop,
                    edge_dim=self.edge_attr_dim,
                    residual=residual_connection
                )
            else:
                conv = GATv2Conv(
                    in_channels=prev_dim, 
                    out_channels=out_dim,
                    heads=num_heads, 
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=self_loop,
                    residual=residual_connection
                )

            layers.append(conv)
            next_dim = out_dim * num_heads if concat else out_dim

            if i < len(layer_specs) - 1:
                layers.append(nn.ReLU())
                
            prev_dim = next_dim
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, data, return_attention: bool = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if (self.use_edge_attr and hasattr(data, "edge_attr")) else None
        attention_by_layer = []  # list of tuples: (edge_index, alpha) per attention layer

        for i, layer in enumerate(self.layers):
            if isinstance(layer, (GATv2Conv)):
                if return_attention:
                    if edge_attr is not None and self.edge_attr_dim is not None:
                        x, (ei_out, alpha) = layer(
                            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
                        )
                    else:
                        x, (ei_out, alpha) = layer(
                            x, edge_index, return_attention_weights=True
                        )
                    attention_by_layer.append({"layer": i, "edge_index": ei_out, "alpha": alpha})
                else:
                    if edge_attr is not None and self.edge_attr_dim is not None:
                        x = layer(x, edge_index, edge_attr=edge_attr)
                    else:
                        x = layer(x, edge_index)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
        
        # Reshape to [batch_size, num_nodes, output_dim]
        batch_size = batch.max().item() + 1
        num_nodes = x.size(0) // batch_size
        x = x.view(batch_size, num_nodes, -1)
        
        if return_attention:
            return x, attention_by_layer
        else:
            return x
        
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for layer_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = layer_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Accepts [M, F] or [B, M, F] and returns [M, E] or [B, M, E]
        return self.net(x)
