# Imports
import os, sys
import gc
import configparser
import pathlib as p
import numpy as np

module_path = os.path.abspath("/dls/science/groups/i04-1/conor_dev/pandda_nn")
if module_path not in sys.path:
    sys.path.append(module_path)
b = sys.path
sys.path = [module_path] + b

#################################
try:
    import matplotlib
    print("Setting MPL backend")
    matplotlib.use('agg')
    matplotlib.interactive(False)
    print(matplotlib.get_backend())
    from matplotlib import pyplot
    pyplot.style.use('ggplot')
    print(pyplot.get_backend())
except Exception as e:
    print("Errored in mpl setup!")
    print(e)
#################################

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

pd.options.display.max_columns = 999

import torch

import clipper_python as clipper

import torch.nn as nn
import torch.optim as optim

from frag_nn.pytorch.network import ClassifierV5
from frag_nn.pytorch.dataset import EventDataset
from frag_nn.pytorch.dataset import OrthogonalGrid
from frag_nn.pytorch.dataset import GetRandomisedLocation, GetRandomisedRotation
from frag_nn.pytorch.dataset import GetAnnotationClassifier, GetDataRefMove, GetDataRefMoveZ


import frag_nn.constants as c

if __name__ == "__main__":
    # Args
    config_path = "/dls/science/groups/i04-1/conor_dev/pandda_nn/frag_nn/params.ini"
    grid_size = 48
    grid_step = 0.5
    filters = 64

    # Get Config
    conf = configparser.ConfigParser()

    conf.read(config_path)

    ds_conf = conf[c.x_chem_database]

    network_type = "classifier"
    network_version = 5
    dataset_version = 3
    train = "cluster"
    transforms = "rottrans"

    state_dict_dir = "/dls/science/groups/i04-1/conor_dev/pandda_nn/"
    state_dict_file = state_dict_dir + "model_params_{}_{}_{}_{}_{}_{}_{}_{}.pt".format(grid_size,
                                                                                  grid_step,
                                                                                  network_type,
                                                                                  network_version,
                                                                                  dataset_version,
                                                                                  train,
                                                                                  transforms,
                                                                                     filters)
    output_file = state_dict_dir + "output_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(grid_size,
                                                                                  grid_step,
                                                                                  network_type,
                                                                                  network_version,
                                                                                  dataset_version,
                                                                                  train,
                                                                                  transforms,
                                                                            filters)

    # Load Database
    events_train = pd.read_csv("/dls/science/groups/i04-1/conor_dev/pandda_nn/new_events_train.csv")

    # Create Dataset

    grid = OrthogonalGrid(grid_size,
                          grid_step)

    dataset_train = EventDataset(events=events_train,
                                 transforms_record=[GetRandomisedLocation(base_trans_max=4.0, secondary_trans_max=0.0),
                                                    GetRandomisedRotation(max_rot=0.0)],
                                 get_annotation=GetAnnotationClassifier(),
                                 get_data=GetDataRefMoveZ(grid)
                                 )

    # Create Dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=32,
                                                   shuffle=True,
                                                   num_workers=16)

    # Define Model

    model = ClassifierV5(filters,
                         grid_dimension=grid_size)

    try:
        model.load_state_dict(torch.load(state_dict_file))
    except Exception as e:
        print(e)

    print(model)

    # Define loss function
    criterion = nn.BCELoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=0.0001)

    # Fit Model

    num_epochs = 500
    running_loss = 0

    print("Beginning training")
    for epoch in range(num_epochs):
        print("Beginning epoch: {}".format(epoch))
        for i, data in enumerate(train_dataloader):
            print("Epoch: {}; iteration: {}".format(epoch, i))
            # get the inputs; data is a list of [inputs, labels]
            x = data["data"]
            y = data["annotation"]
            y = y.view(-1, 2)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # RECORD LOSS
            running_loss += loss.item()

            # print statistics per epoch
            if i % 30 == 29:  # print every 100 mini-batches
                f = open(output_file, "a")
                f.write("Loss at epoch {}, iteration {} is {}".format(epoch,
                                                                    i,
                                                                    running_loss / i) + "\n")
                f.write("{}".format([x for x in outputs]) + "\n")
                f.write("{}".format([x for x in y]) + "\n")
                f.write("#################################################" + "\n")
                f.close()

            if i % 100 == 99:  # print every 100 mini-batches
                f = open(output_file, "a")
                f.write("Checkpointing model" + "\n")
                torch.save(model.state_dict(), state_dict_file)
                f.close()

        f = open(output_file, "a")
        f.write("###################################" + "\n")
        f.write("Loss for epoch {}: {}".format(epoch, running_loss) + "\n")
        f.write(str(outputs) + "\n")
        f.write(str(y) + "\n")
        f.write(str(loss) + "\n")
        running_loss = 0.0
        f.close()

        torch.save(model.state_dict(), state_dict_file)
