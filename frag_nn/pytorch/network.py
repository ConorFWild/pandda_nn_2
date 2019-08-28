# Get imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define Network
class GNINA_regressor(nn.Module):
    """
    CNN for recognising interesting ligands
    """

    def __init__(self, filters, grid_dimension=32, training=False):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.layer_1 = self.block(1, filters)
        self.layer_2 = self.block(filters, filters*2)
        self.layer_3 = self.block(filters*2, filters*4)

        size_after_filters = 4*filters*self.grid_size**3/8**3
        print(size_after_filters)

        self.fc1 = nn.Sequential(nn.Linear(size_after_filters, int(size_after_filters/2)),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(int(size_after_filters/2), 1))

        self.act = nn.Sigmoid()


    def block(self, in_dim, out_dim):

        ops = []
        ops.append(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
        ops.append(nn.ReLU())
        ops.append(nn.MaxPool3d(kernel_size=2, stride=2))
        # ops.append(nn.Dropout3d())

        layer = nn.Sequential(*ops)

        return layer

    def forward(self, x):

        # print(x.shape)

        x = self.layer_1(x)
        # print(x.shape)

        x = self.layer_2(x)
        # print(x.shape)

        x = self.layer_3(x)
        # print(x.shape)

        # print(int(self.filters/2*self.grid_size**3/8**3))
        # x = x.view(-1, int(self.filters/2*self.grid_size**3/8**3))
        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        # x = torch.flatten(x, start_dim=1)
        # print(x.shape)

        x = self.fc1(x)

        # x = F.Dropout(x, training=self.training)

        x = self.fc2(x)
        # print(x.shape)
        # print(x)
        # print(x.shape)

        return self.act(x)

# Define Network
class GNINA_regressor_v2(nn.Module):
    """
    CNN for recognising interesting ligands
    """

    def __init__(self, filters, grid_dimension=32, training=False):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.layer_1 = self.block(1, filters)
        self.layer_2 = self.block(filters, filters*2)
        self.layer_3 = self.block(filters*2, filters*4)
        self.layer_4 = self.block(filters*4, filters*8)

        s1 = self.size_after_layer(1, filters)
        s2 = self.size_after_layer(2, filters*2)
        s3 = self.size_after_layer(3, filters*4)
        s4 = self.size_after_layer(4, filters*8)

        size_after_filters = s4
        print(size_after_filters)

        self.fc1 = nn.Sequential(nn.Linear(size_after_filters, int(size_after_filters/2)),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(int(size_after_filters/2), 1))

        self.act = nn.Sigmoid()

    def size_after_layer(self, n, n_filters):
        return n_filters * ((self.grid_size / 2**n) ** 3)


    def block(self, in_dim, out_dim):

        ops = []
        ops.append(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
        ops.append(nn.ReLU())
        ops.append(nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        ops.append(nn.ReLU())
        ops.append(nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        ops.append(nn.ReLU())
        ops.append(nn.MaxPool3d(kernel_size=2, stride=2))
        # ops.append(nn.Dropout3d())

        layer = nn.Sequential(*ops)

        return layer

    def forward(self, x):

        # print(x.shape)

        x = self.layer_1(x)
        # print(x_1.shape)

        x = self.layer_2(x)
        # print("x2", x_2.shape)

        x = self.layer_3(x)
        # print("x3", x_3.shape)

        x = self.layer_4(x)
        # print("x4", x_4.shape)

        # print(int(self.filters/2*self.grid_size**3/8**3))
        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))
        # x_1 = x_1.view(-1, (x_1.shape[1]*x_1.shape[2]*x_1.shape[3]*x_1.shape[4]))
        # x_2 = x_2.view(-1, (x_2.shape[1]*x_2.shape[2]*x_2.shape[3]*x_2.shape[4]))
        # x_3 = x_3.view(-1, (x_3.shape[1]*x_3.shape[2]*x_3.shape[3]*x_3.shape[4]))
        # x_4 = x_4.view(-1, (x_4.shape[1]*x_4.shape[2]*x_4.shape[3]*x_4.shape[4]))

        # print(x_1.shape)
        # print(x_2.shape)
        # print(x_3.shape)
        # print(x_4.shape)


        # x = torch.flatten(x, start_dim=1)
        # print(x.shape)

        # x_flat = torch.cat([x_3, x_4],
        #                    dim=1)

        x = self.fc1(x)

        # x = F.Dropout(x, training=self.training)

        x = self.fc2(x)
        # print(x.shape)
        # print(x)
        # print(x.shape)

        return self.act(x)




# Define Network
class GNINA_regressor_v3(nn.Module):
    """
    CNN for recognising interesting ligands
    """

    def __init__(self, filters, grid_dimension=32, training=False):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.layer_1 = self.block(1, filters)
        self.layer_2 = self.block(filters, filters*2)
        self.layer_3 = self.block(filters*2, filters*4)
        self.layer_4 = self.block(filters*4, filters*8)

        size_after_filters = self.size_after_layer(4, filters*8)

        self.fc1 = nn.Sequential(nn.Linear(size_after_filters, int(size_after_filters/2)),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(int(size_after_filters/2), 6),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(6, 1))

        self.act = nn.Sigmoid()

    def size_after_layer(self, n, n_filters):
        return n_filters * ((self.grid_size / 2**n) ** 3)

    def block(self, in_dim, out_dim):

        ops = []
        ops.append(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
        ops.append(nn.ReLU())
        ops.append(nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        ops.append(nn.ReLU())
        ops.append(nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        ops.append(nn.ReLU())
        ops.append(nn.MaxPool3d(kernel_size=2, stride=2))
        ops.append(nn.Dropout3d())

        layer = nn.Sequential(*ops)

        return layer

    def forward(self, x):

        x = self.layer_1(x)

        x = self.layer_2(x)

        x = self.layer_3(x)

        x = self.layer_4(x)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        x = self.fc3(x)

        return self.act(x)




class ResidualBlock(nn.Module):

    def __init__(self,filters_out):
        # Instatiate Network layers

        super().__init__()

        self.conv_1 = nn.Conv3d(filters_out, filters_out, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm3d(filters_out)

        self.act = nn.ReLU()

    def forward(self, x):

        residual = self.conv_1(x)
        residual = self.bn(residual)

        x = x + residual

        return self.act(x)


class ResidualLayer(nn.Module):

    def __init__(self, filters_in, filters_out):
        # Instatiate Network layers

        super().__init__()

        self.conv_1 = nn.Sequential(nn.Conv3d(filters_in, filters_out, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(filters_out),
                                    nn.ReLU())

        self.res_1 = ResidualBlock(filters_out)
        # self.res_2 = ResidualBlock(filters_out)

        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

        self.drop = nn.Dropout3d()

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.res_1(x)
        # x = self.res_2(x)
        x = self.mp(x)

        x = self.drop(x)

        return self.act(x)


class GNINA_regressor_v4(nn.Module):
    """
    CNN for recognising interesting ligands
    """

    def __init__(self, filters, grid_dimension=32, training=False):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.conv_1 = self.fc1 = nn.Sequential(nn.Conv3d(1, filters, kernel_size=3, stride=1, padding=1),
                                               nn.BatchNorm3d(filters),
                                               nn.ReLU())

        self.layer_1 = ResidualLayer(filters, filters*2)
        self.layer_2 = ResidualLayer(filters*2, filters*4)
        self.layer_3 = ResidualLayer(filters*4, filters*8)

        print(int(grid_dimension/2**3))
        self.avp3 = nn.AvgPool3d(kernel_size=int(grid_dimension/2**3), stride=int(grid_dimension/2**3), padding=1)

        self.fc1 = nn.Sequential(nn.Linear(filters*8, filters*4),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(filters * 4, 1))

        self.act = nn.Sigmoid()

    def forward(self, x):

        x = self.conv_1(x)

        x = self.layer_1(x)

        x = self.layer_2(x)

        x = self.layer_3(x)
        # print(x.shape)

        x = self.avp3(x)
        # print(x.shape)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)


class ResidualLayerNoDrop(nn.Module):

    def __init__(self, filters_in, filters_out):
        # Instatiate Network layers

        super().__init__()

        self.conv_1 = nn.Sequential(nn.Conv3d(filters_in, filters_out, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(filters_out),
                                 nn.ReLU())

        self.res_1 = ResidualBlock(filters_out)
        # self.res_2 = ResidualBlock(filters_out)

        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.res_1(x)
        # x = self.res_2(x)
        x = self.mp(x)

        return self.act(x)


class GNINA_regressor_v5(nn.Module):
    """
    CNN for recognising interesting ligands
    1x introductory convolution, 3x residual layers of 2, 2x
    No dropout
    No global pooling
    """

    def __init__(self, filters, grid_dimension=32, training=False):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.conv_1 = nn.Sequential(nn.Conv3d(1, filters, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(filters),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer_1 = ResidualLayerNoDrop(filters, filters * 2)
        self.layer_2 = ResidualLayerNoDrop(filters*2, filters*4)
        self.layer_3 = ResidualLayerNoDrop(filters*4, filters*8)

        # print(int(grid_dimension/2**3))
        # self.avp3 = nn.AvgPool3d(kernel_size=int(grid_dimension/2**3), stride=int(grid_dimension/2**3), padding=1)

        size_after_convs = int((grid_dimension/(2**4))**3)
        n_in = filters*8*size_after_convs
        print(n_in)

        self.fc1 = nn.Sequential(nn.Linear(n_in, filters*2),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(filters * 2, 1))

        self.act = nn.Sigmoid()

    def forward(self, x):

        x = self.conv_1(x)
        # print(x.shape)

        x = self.layer_1(x)
        # print(x.shape)

        x = self.layer_2(x)
        # print(x.shape)

        x = self.layer_3(x)
        # print(x.shape)

        # x = self.avp3(x)
        # print(x.shape)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)


class GNINA_regressor_v6(nn.Module):
    """
    CNN for recognising interesting ligands
    1x introductory convolution, 3x residual layers of 2, 2x
    No dropout
    No global pooling
    """

    def __init__(self, filters, grid_dimension=32, training=False):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.conv_1 = nn.Sequential(nn.Conv3d(2, filters, kernel_size=3, stride=1, padding=1),
                                               nn.BatchNorm3d(filters),
                                               nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer_1 = ResidualLayerNoDrop(filters, filters*2)
        self.layer_2 = ResidualLayerNoDrop(filters*2, filters*4)
        self.layer_3 = ResidualLayerNoDrop(filters*4, filters*8)

        # print(int(grid_dimension/2**3))
        # self.avp3 = nn.AvgPool3d(kernel_size=int(grid_dimension/2**3), stride=int(grid_dimension/2**3), padding=1)

        size_after_convs = int((grid_dimension/(2**4))**3)
        n_in = filters*8*size_after_convs
        print(n_in)

        self.fc1 = nn.Sequential(nn.Linear(n_in, filters*2),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(filters * 2, 1))

        self.act = nn.Sigmoid()

    def forward(self, x):

        x = self.conv_1(x)
        # print(x.shape)

        x = self.layer_1(x)
        # print(x.shape)

        x = self.layer_2(x)
        # print(x.shape)

        x = self.layer_3(x)
        # print(x.shape)

        # x = self.avp3(x)
        # print(x.shape)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)


class ResidualLayerWithDrop(nn.Module):

    def __init__(self, filters_in, filters_out, training=True):
        # Instatiate Network layers

        super().__init__()

        self.conv_1 = nn.Sequential(nn.Conv3d(filters_in, filters_out, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(filters_out),
                                 nn.ReLU())

        self.res_1 = ResidualBlock(filters_out)
        # self.res_2 = ResidualBlock(filters_out)

        self.drop = nn.Dropout3d(p=0.5)

        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)
        self.do_drop = training

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.res_1(x)
        # x = self.res_2(x)
        if self.training:
            x = self.drop(x)
        x = self.mp(x)

        return self.act(x)


class GNINA_regressor_v7(nn.Module):
    """
    CNN for recognising interesting ligands
    1x introductory convolution, 3x residual layers of 2, 2x
    No dropout
    No global pooling
    """

    def __init__(self, filters, grid_dimension=32, training=False, do_drop=True):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.conv_1 = nn.Sequential(nn.Conv3d(2, filters, kernel_size=3, stride=1, padding=1),
                                               nn.BatchNorm3d(filters),
                                               nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer_1 = ResidualLayerWithDrop(filters, filters*2, do_drop)
        self.layer_2 = ResidualLayerWithDrop(filters*2, filters*4, do_drop)
        self.layer_3 = ResidualLayerWithDrop(filters*4, filters*8, do_drop)

        # print(int(grid_dimension/2**3))
        # self.avp3 = nn.AvgPool3d(kernel_size=int(grid_dimension/2**3), stride=int(grid_dimension/2**3), padding=1)

        size_after_convs = int((grid_dimension/(2**4))**3)
        n_in = filters*8*size_after_convs
        print(n_in)

        self.fc1 = nn.Sequential(nn.Linear(n_in, filters*2),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(filters * 2, 1))

        self.act = nn.Sigmoid()

    def forward(self, x):

        x = self.conv_1(x)
        # print(x.shape)

        x = self.layer_1(x)
        # print(x.shape)

        x = self.layer_2(x)
        # print(x.shape)

        x = self.layer_3(x)
        # print(x.shape)

        # x = self.avp3(x)
        # print(x.shape)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)


class ResidualLayerWithDropx2(nn.Module):

    def __init__(self, filters_in, filters_out, do_drop=True):
        # Instatiate Network layers

        super().__init__()

        self.conv_1 = nn.Sequential(nn.Conv3d(filters_in, filters_out, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(filters_out),
                                 nn.ReLU())

        self.res_1 = ResidualBlock(filters_out)
        self.res_2 = ResidualBlock(filters_out)
        self.res_3 = ResidualBlock(filters_out)

        # self.res_2 = ResidualBlock(filters_out)

        self.drop = nn.Dropout3d(p=0.1)

        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)
        self.do_drop = do_drop

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)


        if self.do_drop:
            x = self.drop(x)
        x = self.mp(x)

        return self.act(x)


class GNINA_regressor_v8(nn.Module):
    """
    CNN for recognising interesting ligands
    1x introductory convolution, 3x residual layers of 2, 2x
    No dropout
    No global pooling
    """

    def __init__(self, filters, grid_dimension=32, training=False, do_drop=True):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.conv_1 = nn.Sequential(nn.Conv3d(2, filters, kernel_size=3, stride=1, padding=1),
                                               nn.BatchNorm3d(filters),
                                               nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer_1 = ResidualLayerWithDropx2(filters, filters*2, do_drop)
        self.layer_2 = ResidualLayerWithDropx2(filters*2, filters*4, do_drop)
        self.layer_3 = ResidualLayerWithDropx2(filters*4, filters*8, do_drop)

        # print(int(grid_dimension/2**3))
        # self.avp3 = nn.AvgPool3d(kernel_size=int(grid_dimension/2**3), stride=int(grid_dimension/2**3), padding=1)

        size_after_convs = int((grid_dimension/(2**4))**3)
        n_in = filters*8*size_after_convs
        print(n_in)

        self.fc1 = nn.Sequential(nn.Linear(n_in, filters*2),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(filters * 2, 1))

        self.act = nn.Sigmoid()

    def forward(self, x):

        x = self.conv_1(x)
        # print(x.shape)

        x = self.layer_1(x)
        # print(x.shape)

        x = self.layer_2(x)
        # print(x.shape)

        x = self.layer_3(x)
        # print(x.shape)

        # x = self.avp3(x)
        # print(x.shape)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)


class GNINA_regressor_v9(nn.Module):
    """
    CNN for recognising interesting ligands
    """

    def __init__(self, filters, grid_dimension=32, training=False):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.layer_1 = nn.Sequential(nn.Conv3d(2, filters, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters),
                                     nn.ReLU(),
                                     nn.Conv3d(filters, filters, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2),
                                     nn.Dropout3d(p=0.1))
        self.layer_2 = nn.Sequential(nn.Conv3d(filters, filters*2, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters*2),
                                     nn.ReLU(),
                                     nn.Conv3d(filters, filters, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2),
                                     nn.Dropout3d(p=0.1))
        self.layer_3 = nn.Sequential(nn.Conv3d(filters*2, filters*4, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters*4),
                                     nn.ReLU(),
                                     nn.Conv3d(filters, filters, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2),
                                     nn.Dropout3d(p=0.1))
        self.layer_4 = nn.Sequential(nn.Conv3d(filters*4, filters*8, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters*8),
                                     nn.ReLU(),
                                     nn.Conv3d(filters, filters, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2),
                                     nn.Dropout3d(p=0.1))
        # self.layer_5 = nn.Sequential(nn.Conv3d(filters*8, filters*16, kernel_size=3, stride=1, padding=1),
        #                              nn.BatchNorm3d(filters*16),
        #                              nn.ReLU(),
        #                              nn.MaxPool3d(kernel_size=2, stride=2),
        #                              nn.Dropout3d(p=0.1))

        size_after_convs = int((grid_dimension / (2 ** 4)) ** 3)
        n_in = filters * 8 * size_after_convs


        self.fc1 = nn.Sequential(nn.Linear(n_in, int(n_in/2)),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(int(n_in/2), 1))

        self.act = nn.Sigmoid()

    def forward(self, x):

        # print(x.shape)

        x = self.layer_1(x)
        # print(x.shape)

        x = self.layer_2(x)
        # print(x.shape)

        x = self.layer_3(x)
        # print(x.shape)

        x = self.layer_4(x)

        # x = self.layer_5(x)



        # print(int(self.filters/2*self.grid_size**3/8**3))
        # x = x.view(-1, int(self.filters/2*self.grid_size**3/8**3))
        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        # x = torch.flatten(x, start_dim=1)
        # print(x.shape)

        x = self.fc1(x)

        # x = F.Dropout(x, training=self.training)

        x = self.fc2(x)
        # print(x.shape)
        # print(x)
        # print(x.shape)

        return self.act(x)


class GNINA_regressor_v10(nn.Module):
    """
    CNN for recognising interesting ligands
    """

    def __init__(self, filters, grid_dimension=32, training=False):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.layer_1 = nn.Sequential(nn.Conv3d(2, filters, kernel_size=9, stride=1, padding=4),
                                     nn.BatchNorm3d(filters),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer_2 = nn.Sequential(nn.Conv3d(filters, filters*2, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters*2),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer_3 = nn.Sequential(nn.Conv3d(filters*2, filters*4, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters*4),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer_4 = nn.Sequential(nn.Conv3d(filters*4, filters*8, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters*8),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2))

        size_after_convs = int((grid_dimension / (2 ** 4)) ** 3)
        n_in = filters * 8 * size_after_convs

        self.fc1 = nn.Sequential(nn.Linear(n_in, int(n_in/2)),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(int(n_in/2), 1))

        self.act = nn.Sigmoid()

    def forward(self, x):

        # print(x.shape)
        print(x.min(), x.max())

        x = self.layer_1(x)
        # print(x.shape)

        x = self.layer_2(x)
        # print(x.shape)

        x = self.layer_3(x)
        # print(x.shape)

        x = self.layer_4(x)
        # print(x.shape)


        # x = self.layer_5(x)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        # x = torch.flatten(x, start_dim=1)
        # print(x.shape)

        x = self.fc1(x)

        # x = F.Dropout(x, training=self.training)

        x = self.fc2(x)
        # print(x.shape)
        # print(x)
        # print(x.shape)

        return self.act(x)


class ClassifierV1(nn.Module):
    """
    CNN for recognising interesting ligands
    """

    def __init__(self, filters, grid_dimension=32, training=False):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.layer_1 = nn.Sequential(nn.Conv3d(2, filters, kernel_size=9, stride=1, padding=4),
                                     nn.BatchNorm3d(filters),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer_2 = nn.Sequential(nn.Conv3d(filters, filters*2, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters*2),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer_3 = nn.Sequential(nn.Conv3d(filters*2, filters*4, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters*4),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer_4 = nn.Sequential(nn.Conv3d(filters*4, filters*8, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(filters*8),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=2, stride=2))

        size_after_convs = int((grid_dimension / (2 ** 4)) ** 3)
        n_in = filters * 8 * size_after_convs

        self.fc1 = nn.Sequential(nn.Linear(n_in, int(n_in/2)),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(int(n_in/2), 2))

        self.act = nn.Softmax()

    def forward(self, x):

        # print(x.shape)
        # print(x.min(), x.max())

        x = self.layer_1(x)
        # print(x.shape)

        x = self.layer_2(x)
        # print(x.shape)

        x = self.layer_3(x)
        # print(x.shape)

        x = self.layer_4(x)
        # print(x.shape)


        # x = self.layer_5(x)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        # x = torch.flatten(x, start_dim=1)
        # print(x.shape)

        x = self.fc1(x)

        # x = F.Dropout(x, training=self.training)

        x = self.fc2(x)
        # print(x.shape)
        # print(x)
        # print(x.shape)

        return self.act(x)


class ClassifierV2(nn.Module):
    """
    CNN for recognising interesting ligands
    1x introductory convolution, 3x residual layers of 2, 2x
    No dropout
    No global pooling
    """

    def __init__(self, filters, grid_dimension=32, training=True):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.conv_1 = nn.Sequential(nn.Conv3d(2, filters, kernel_size=9, stride=1, padding=4),
                                    nn.BatchNorm3d(filters),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer_1 = ResidualLayerWithDrop(filters, filters * 2, training)
        self.layer_2 = ResidualLayerWithDrop(filters*2, filters*4, training)
        self.layer_3 = ResidualLayerWithDrop(filters*4, filters*8, training)


        size_after_convs = int((grid_dimension/(2**4))**3)
        n_in = filters*8*size_after_convs

        self.fc1 = nn.Sequential(nn.Linear(n_in, filters*2),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(filters * 2, 2))

        self.act = nn.Softmax()

    def forward(self, x):

        x = self.conv_1(x)

        x = self.layer_1(x)

        x = self.layer_2(x)

        x = self.layer_3(x)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)


class ClassifierV3(nn.Module):
    """
    CNN for recognising interesting ligands
    1x introductory convolution, 3x residual layers of 2, 2x
    No dropout
    No global pooling
    """

    def __init__(self, filters, grid_dimension=32, training=True):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.conv_1 = nn.Sequential(nn.Conv3d(3, filters, kernel_size=9, stride=1, padding=4),
                                    nn.BatchNorm3d(filters),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer_1 = ResidualLayerWithDrop(filters, filters * 2, training)
        self.layer_2 = ResidualLayerWithDrop(filters*2, filters*4, training)
        self.layer_3 = ResidualLayerWithDrop(filters*4, filters*8, training)


        size_after_convs = int((grid_dimension/(2**4))**3)
        n_in = filters*8*size_after_convs

        self.fc1 = nn.Sequential(nn.Linear(n_in, filters*2),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(filters * 2, 2))

        self.act = nn.Softmax()

    def forward(self, x):

        x = self.conv_1(x)

        x = self.layer_1(x)

        x = self.layer_2(x)

        x = self.layer_3(x)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)


class ClassifierV4(nn.Module):
    """
    CNN for recognising interesting ligands
    1x introductory convolution, 3x residual layers of 2, 2x
    No dropout
    No global pooling
    """

    def __init__(self, filters, grid_dimension=32, training=True):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.bn = nn.BatchNorm3d(3)

        self.conv_1 = nn.Sequential(nn.Conv3d(3, filters, kernel_size=9, stride=1, padding=4),
                                    nn.BatchNorm3d(filters),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer_1 = ResidualLayerWithDrop(filters, filters * 2, training)
        self.layer_2 = ResidualLayerWithDrop(filters*2, filters*4, training)
        self.layer_3 = ResidualLayerWithDrop(filters*4, filters*8, training)


        size_after_convs = int((grid_dimension/(2**4))**3)
        n_in = filters*8*size_after_convs

        self.fc1 = nn.Sequential(nn.Linear(n_in, filters*2),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(filters * 2, 2))

        self.act = nn.Softmax()

    def forward(self, x):

        x = self.bn(x)

        x = self.conv_1(x)

        x = self.layer_1(x)

        x = self.layer_2(x)

        x = self.layer_3(x)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)


class ClassifierV5(nn.Module):
    """
    CNN for recognising interesting ligands
    1x introductory convolution, 3x residual layers of 2, 2x
    No dropout
    No global pooling
    """

    def __init__(self, filters, grid_dimension=32, training=True):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.bn = nn.BatchNorm3d(3)

        self.conv_1 = nn.Sequential(nn.Conv3d(3, filters, kernel_size=9, stride=1, padding=4),
                                    nn.BatchNorm3d(filters),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer_1 = ResidualLayerWithDrop(filters, filters * 2, training)
        self.layer_2 = ResidualLayerWithDrop(filters*2, filters*4, training)
        self.layer_3 = ResidualLayerWithDrop(filters*4, filters*8, training)

        size_after_convs = int((grid_dimension/(2**4))**3)
        n_in = filters*8*size_after_convs

        self.fc1 = nn.Sequential(nn.Linear(n_in, int(n_in/4)),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(int(n_in/4), 2))

        self.act = nn.Softmax()

    def forward(self, x):

        x = self.bn(x)

        x = self.conv_1(x)

        x = self.layer_1(x)

        x = self.layer_2(x)

        x = self.layer_3(x)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)


class ClassifierV6(nn.Module):
    """
    CNN for recognising interesting ligands
    1x introductory convolution, 3x residual layers of 2, 2x
    No dropout
    No global pooling
    """

    def __init__(self, filters, grid_dimension=32, training=True):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_dimension

        self.conv_pool_1 = nn.Sequential(nn.Conv3d(3, filters, kernel_size=9, stride=1, padding=4),
                                         nn.BatchNorm3d(filters),
                                         nn.MaxPool3d(kernel_size=2, stride=2),
                                         nn.ReLU(),
                                         nn.Dropout3d(p=0.5)
                                         )

        self.conv_pool_2 = nn.Sequential(nn.Conv3d(filters, filters * 2, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm3d(filters),
                                         nn.MaxPool3d(kernel_size=2, stride=2),
                                         nn.ReLU(),
                                         nn.Dropout3d(p=0.5)
                                         )

        self.conv_pool_3 = nn.Sequential(nn.Conv3d(filters * 2, filters * 4, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm3d(filters),
                                         nn.MaxPool3d(kernel_size=2, stride=2),
                                         nn.ReLU(),
                                         nn.Dropout3d(p=0.5)
                                         )

        size_after_convs = int((grid_dimension/(2**3))**3)
        n_in = filters*4*size_after_convs

        self.fc1 = nn.Sequential(nn.Linear(n_in, int(n_in/4)),
                                           nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(int(n_in/4), 2))

        self.act = nn.Softmax()

    def forward(self, x):

        x = self.conv_pool_1(x)

        x = self.conv_pool_2(x)

        x = self.conv_pool_3(x)

        x = x.view(-1, (x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)
