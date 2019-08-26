# Get imports

import tensorflow as tf

class GNINA(tf.keras.Model):

    def __init__(self, grid_dimension=10):

        # TODO: This breaking for some reason
        super().__init__()


        # self.latent_dim = latent_dim
        self.net = tf.keras.Sequential()

        self.net.add(tf.keras.layers.InputLayer(input_shape=(grid_dimension, grid_dimension, grid_dimension, 1)))

        self.conv_mpool_drop(self.net, 32)

        self.conv_mpool_drop(self.net, 64)

        self.conv_mpool_drop(self.net, 128)


        self.net.add(tf.keras.layers.Flatten())

        self.net.add(tf.keras.layers.Dense(1))

    def conv_mpool_drop(self, net, filters, ksize=2, kstride_conv=(1, 1, 1), kstrid_pool=(2,2,2), activation=tf.nn.relu, drop_rate=0.3):
        net.add(tf.keras.layers.Conv3D(
            filters=filters, kernel_size=ksize, strides=kstride_conv, activation=activation, padding="same"))
        net.add(tf.keras.layers.MaxPooling3D((ksize, ksize, ksize), strides=kstrid_pool, padding="same"))
        net.add(tf.keras.layers.Dropout(drop_rate))


    def call(self, x):
        y_pred = self.net(x)
        print(y_pred)
        return y_pred


class GNINA_categorical(tf.keras.Model):

    def __init__(self, grid_dimension=10):

        # TODO: This breaking for some reason
        super().__init__()


        # self.latent_dim = latent_dim
        self.net = tf.keras.Sequential()

        self.net.add(tf.keras.layers.InputLayer(input_shape=(grid_dimension, grid_dimension, grid_dimension, 1)))

        self.conv_mpool_drop(self.net, 32)

        self.conv_mpool_drop(self.net, 64)

        self.conv_mpool_drop(self.net, 128)


        self.net.add(tf.keras.layers.Flatten())

        self.net.add(tf.keras.layers.Dense(128, activation="relu"))

        self.net.add(tf.keras.layers.Dense(64, activation="relu"))

        self.net.add(tf.keras.layers.Dense(2, activation='softmax'))


    def conv_mpool_drop(self, net, filters, ksize=2, kstride_conv=(1, 1, 1), kstrid_pool=(2,2,2), activation=tf.nn.relu, drop_rate=0.3):
        net.add(tf.keras.layers.Conv3D(
            filters=filters, kernel_size=ksize, strides=kstride_conv, activation=activation, padding="same"))
        net.add(tf.keras.layers.MaxPooling3D((ksize, ksize, ksize), strides=kstrid_pool, padding="same"))
        net.add(tf.keras.layers.Dropout(drop_rate))


    def call(self, x):
        y_pred = self.net(x)
        print(y_pred)
        return y_pred


# Define Network
#
# class FragmentNet(nn.module):
#     """
#     CNN for recognising interesting ligands
#     """
#
#     def __init__(self):
#         # Instatiate Network layers
#
#         super(FragmentNet, self).__init__()
#
#         self.conv1 = nn.Conv3d(1, 10, kernel_size=3)
#         self.drop1 = nn.Dropout3d()
#
#         self.conv1 = nn.Conv3d(1, 10, kernel_size=3)
#         self.drop1 = nn.Dropout3d()
#
#         self.conv1 = nn.Conv3d(1, 10, kernel_size=3)
#         self.drop1 = nn.Dropout3d()
#
#         self.fc1 = nn.Linear()
#
#         self.fc2 = nn.Linear()
#
#
# def forward(self, x):
#
#         x = F.relu(F.max_pool3d(self.drop1(self.conv1(x)), 2))
#
#         x = F.relu(F.max_pool3d(self.drop1(self.conv1(x)), 2))
#
#         x = F.relu(F.max_pool3d(self.drop1(self.conv1(x)), 2))
#
#         x = x.view(-1, 320)
#
#         x = F.relu(self.fc1(x))
#
#         x = F.Dropout(x, training=self.training)
#
#         x = F.relu(self.fc2(x))
#
#         return F.log_softmax(x, dim=1)