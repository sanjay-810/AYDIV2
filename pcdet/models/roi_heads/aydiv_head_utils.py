import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, in_channel=9, out_channels=32):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channel, out_channels, 1)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.transpose(1,2)

        return x

class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.depth_in, self.valid_in = channels
        middle = self.valid_in // 4
        self.fc1 = nn.Linear(self.depth_in, middle)
        self.fc2 = nn.Linear(self.valid_in, middle)
        self.fc3 = nn.Linear(2*middle, 2)
        self.conv1 = nn.Sequential(nn.Conv1d(self.depth_in, self.valid_in, 1),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(self.valid_in, self.valid_in, 1),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())

    def forward(self, depth_feas, valid_feas):
        batch = depth_feas.size(0)

        depth_feas_f = depth_feas.transpose(1,2).contiguous().view(-1, self.depth_in)
        valid_feas_f = valid_feas.transpose(1,2).contiguous().view(-1, self.valid_in)

        depth_feas_f_ = self.fc1(depth_feas_f)
        valid_feas_f_ = self.fc2(valid_feas_f)
        depth_valid_feas_f = torch.cat([depth_feas_f_, valid_feas_f_],dim=-1)
        weight = torch.sigmoid(self.fc3(depth_valid_feas_f))

        depth_weight = weight[:,0].squeeze()
        depth_weight = depth_weight.view(batch, 1, -1)

        valid_weight = weight[:,1].squeeze()
        valid_weight = valid_weight.view(batch, 1, -1)

        depth_features_att = self.conv1(depth_feas)  * depth_weight
        valid_features_att     =  self.conv2(valid_feas)      *  valid_weight

        return depth_features_att, valid_features_att

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_fn=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim

        for hidden_layer_size in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_layer_size))
            layers.append(activation_fn)
            last_dim = hidden_layer_size
        layers.append(nn.Linear(last_dim, output_dim))k
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class local(nn.Module):
    def __init__(self, image_dim, output_features_dim):
        super(local, self).__init__()
        self.mlp = MLP(image_dim, 3, image_dim)
        self.query_layer = nn.Linear(image_dim, output_features_dim)
        self.key_layer = nn.Linear(image_dim, output_features_dim)
        self.value_layer = nn.Linear(image_dim, output_features_dim)

    def forward(self,image):
        query = self.query_layer(self.mlp(image))
        key = self.key_layer(self.mlp(image))
        value = self.value_layer(self.mlp(image))
        query = nn.LayerNorm(query)
        key = nn.LayerNorm(key)
        value = nn.LayerNorm(value)
        attention_scores = torch.matmul(query, key.transpose(2, 1))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_probs, value)
        return attended_features
    
class GCFAT(nn.Module):
    def __init__(self, image_dim, depth_dim, d_model, num_heads):
        super(GCFAT, self).__init__() 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.lmsa = local(image_dim,image_dim)
        self.mlp = MLP(image_dim, 3, image_dim)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)       
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)
    def forward(self, depth, image, mask=None):
        image = self.lmsa(image)
        query = self.query_layer(self.mlp(depth))
        key = self.key_layer(self.mlp(image))
        value = self.value_layer(self.mlp(image))
        batch_size = query.size(0)
        query = nn.LayerNorm(query)
        key = nn.LayerNorm(key)
        value = nn.LayerNorm(value)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k).float())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        return self.fc(output)
    
    
class Sparse_Attention(nn.Module):
    def __init__(self, image_dim, output_features_dim):
        super(local, self).__init__()
        self.mlp = MLP(image_dim, 3, image_dim)
        self.query_layer = nn.Linear(image_dim, output_features_dim)
        self.key_layer = nn.Linear(image_dim, output_features_dim)
        self.value_layer = nn.Linear(image_dim, output_features_dim)

    def forward(self,lidar,image):
        query = self.query_layer(lidar)
        key = self.key_layer(image)
        value = self.value_layer(image)
        attention_scores = torch.matmul(query, key.transpose(2, 1))
        attention_probs = F.relu(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_probs, value)
        return nn.LayerNorm(attended_features)

class VGA(nn.Module):
    def __init__(self, pseudo_in, valid_in, outplanes):
        super(VGA, self).__init__()
        self.attention = Attention(channels = [pseudo_in, valid_in])
        self.conv1 = torch.nn.Conv1d(valid_in + valid_in, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, pseudo_features, valid_features):
        pseudo_features_att, valid_features_att=  self.attention(pseudo_features, valid_features)
        fusion_features = torch.cat([valid_features_att, pseudo_features_att], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features