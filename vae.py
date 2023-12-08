import torch
import torch.nn as nn
import torch.nn.functional as F
class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        #print ("INPUT DIM: ", input_dim)
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        #print("X: ", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        #print("X: ", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print("X: ", x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        #print("X: ", x.shape)
        x = self.bn4(self.conv4(x))
        #print("X: ", x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        #print("X: ", x.shape)
        x = x.view(-1, 512)
        #print("X: ", x.shape)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        #print("m: ", m.shape)
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        #print("m: ", m.shape)
        m = self.fc3_m(m)
        #print("m: ", m.shape)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        #print("v: ", v.shape)
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        #print("v: ", v.shape)
        v = self.fc3_v(v)
        #print("MEAN", m.shape)
        #print("V: ", v.shape)
        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v
    
class PointNetDecoder(nn.Module):
    def __init__(self, zdim, output_dim=3):
        super().__init__()
        self.zdim = zdim
        self.fc1_up = nn.Linear(zdim, 128)
        self.fc2_up = nn.Linear(128, 256)
        self.fc3_up = nn.Linear(256, 512)
        self.fc_bn1_up = nn.BatchNorm1d(128)
        self.fc_bn2_up = nn.BatchNorm1d(256)

        self.conv4_up = nn.ConvTranspose1d(512, 256, 1)
        self.conv3_up = nn.ConvTranspose1d(256, 128, 1)
        self.conv2_up = nn.ConvTranspose1d(128, 128, 1)
        self.conv1_up = nn.ConvTranspose1d(128, output_dim, 1)

        self.bn4_up = nn.BatchNorm1d(256)
        self.bn3_up = nn.BatchNorm1d(128)
        self.bn2_up = nn.BatchNorm1d(128)

    def forward(self, x):
        # Upsample latent code
        #print("DECODER: ", x.shape)
        x = F.relu(self.fc_bn1_up(self.fc1_up(x)))
        #print("DECODER: ", x.shape)
        x = F.relu(self.fc_bn2_up(self.fc2_up(x)))
        #print("DECODER: ", x.shape)
        x = self.fc3_up(x)
        #print("DECODER: ", x.shape)
        x = x.view(x.shape[0], x.shape[1], -1)
        #print("DECODER: ", x.shape)

        # Deconvolutional layers
        x = F.relu(self.bn4_up(self.conv4_up(x)))
        #print("DECODER: ", x.shape)
        x = F.relu(self.bn3_up(self.conv3_up(x)))
        #print("DECODER: ", x.shape)
        x = F.relu(self.bn2_up(self.conv2_up(x)))
        #print("DECODER: ", x.shape)
        x = self.conv1_up(x)
        #print("DECODER: ", x.shape)
        return x
    
class PointCloudDecoder(nn.Module):
    def __init__(self, zdim, output_dim=3):
        super(PointCloudDecoder, self).__init__()
        self.fc1 = nn.Linear(zdim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 3072)  # 3072 = 1024 * 3
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, latent_code):
        # Upsample latent code
        x = F.relu(self.bn1(self.fc1(latent_code)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        #print("DECODER: ", x.shape)
        pointcloud = x.view(-1, 1024, 3)
        #print("DECODER: ", pointcloud.shape)
        return pointcloud
    
class PointAltDecoder(nn.Module):
    def __init__(self, zdim, output_dim=3):
        super(PointAltDecoder, self).__init__()
        # Encoder layers
        self.fc1 = nn.Linear(zdim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        # Residual block 1
        self.res_block1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        # Skip connection 1
        self.skip_conn1 = nn.Linear(512, 512)
        # Residual block 2
        self.res_block2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        # Decoder layers
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 3072)  # 3072 = 1024 * 3

    def forward(self, latent_code):
        # Upsample latent code
        x = F.relu(self.bn1(self.fc1(latent_code)))
        # Residual block 1
        x = self.res_block1(x)
        # Skip connection 1
        x = x + self.skip_conn1(latent_code)
        # Residual block 2
        x = self.res_block2(x)
        # Upsample to final point cloud size
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        #print("DECODER: ", x.shape)
        pointcloud = x.view(-1, 1024, 3)
        #print("DECODER: ", x.shape)
        return pointcloud

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        #self.encoder = Encoder(input_dim, latent_dim)
        self.encoder = PointNetEncoder(zdim=latent_dim)
        #self.decoder = Decoder(latent_dim, input_dim)
        self.decoder = PointCloudDecoder(zdim=latent_dim)        
        #self.decoder = PointAltDecoder(zdim=latent_dim)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    


# Example usage
#input_dim = 1024 * 3  # Assuming each point cloud has 1000 points with x, y, z coordinates
#latent_dim = 64  # Size of the latent vector

#model = VAE(input_dim, latent_dim)

# Assuming point_cloud is your input tensor of shape [batch_size, 3000]
# point_cloud = ...
# reconstructed, mu, log_var = model(point_cloud)
