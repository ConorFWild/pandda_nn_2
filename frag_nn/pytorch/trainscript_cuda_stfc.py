# Imports
import os, sys
import gc
import configparser
import pathlib as p
import numpy as np

module_path = os.path.abspath("/home/zoh22914/pandda_nn_2")
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

from frag_nn.pytorch.network import ClassifierV6
from frag_nn.pytorch.dataset import EventDataset
from frag_nn.pytorch.dataset import OrthogonalGrid
from frag_nn.pytorch.dataset import GetRandomisedLocation, GetRandomisedRotation, SetRoot
from frag_nn.pytorch.dataset import GetAnnotationClassifier, GetDataRefMove, GetDataRefMoveZ


import frag_nn.constants as c

if __name__ == "__main__":
    # Args
    database_file_string = "/home/zoh22914/pandda_nn_2/new_events_train_no_cheat.csv"
    config_path = "/home/zoh22914/pandda_nn_2/frag_nn/params.ini"
    grid_size = 48
    grid_step = 0.5
    filters = 64

    # Get Config
    conf = configparser.ConfigParser()

    conf.read(config_path)

    ds_conf = conf[c.x_chem_database]

    network_type = "classifier"
    network_version = 6
    dataset_version = 3
    train = "gpu"
    transforms = "rottrans"
    num_epochs = 1


    state_dict_dir = "/home/zoh22914/pandda_nn_2/"
    state_dict_file = state_dict_dir + "model_params_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt".format(grid_size,
                                                                                  grid_step,
                                                                                  network_type,
                                                                                  network_version,
                                                                                  dataset_version,
                                                                                  train,
                                                                                  transforms,
                                                                                     filters,
                                                                                        num_epochs)
    output_file = state_dict_dir + "output_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(grid_size,
                                                                                  grid_step,
                                                                                  network_type,
                                                                                  network_version,
                                                                                  dataset_version,
                                                                                  train,
                                                                                  transforms,
                                                                            filters,
                                                                               num_epochs)

    print("State dict is at: {}".format(state_dict_file))


    # Write out CUDA device
    f = open(output_file, "w")
    f.write("Cuda is available?: {}\n".format(torch.cuda.is_available()))
    f.write("Cuda device is: {}\n".format(torch.cuda.get_device_name(0)))
    f.close()

    # Load Database
    events_train = pd.read_csv(database_file_string)

    # Create Dataset

    grid = OrthogonalGrid(grid_size,
                          grid_step)

    dataset_train = EventDataset(events=events_train,
                                 transforms_record=[GetRandomisedLocation(base_trans_max=4.0, secondary_trans_max=0.0),
                                                    GetRandomisedRotation(max_rot=0.0),
                                                    SetRoot("/data/data")],
                                 get_annotation=GetAnnotationClassifier(),
                                 get_data=GetDataRefMoveZ(grid),
                                 )

    # Create Dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=48)

    # Define Model

    model = ClassifierV6(filters,
                         grid_dimension=grid_size)
    model.cuda()
    model_c = model.to("cuda")

    try:
        model.load_state_dict(torch.load(state_dict_file))
    except Exception as e:
        print(e)

    print(model)

    # Define loss function
    criterion = nn.BCELoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=0.00001)

    # Fit Model

    running_loss = 0

    print("Beginning training")
    for epoch in range(num_epochs):
        print("Beginning epoch: {}".format(epoch))
        for i, data in enumerate(train_dataloader):
            # print("Epoch: {}; iteration: {}".format(epoch, i))
            # get the inputs; data is a list of [inputs, labels]
            x = data["data"]
            y = data["annotation"]
            y = y.view(-1, 2)

            # Set to cuda
            x_c = x.to("cuda")
            y_c = y.to("cuda")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model_c(x_c)
            loss = criterion(outputs, y_c)
            loss.backward()
            optimizer.step()

            # RECORD LOSS
            running_loss += loss.item()

            # print statistics per epoch
            if i % 100 == 99:  # print every 100 mini-batches
                f = open(output_file, "a")
                f.write("Loss at epoch {}, iteration {} is {}".format(epoch,
                                                                    i,
                                                                    running_loss / i) + "\n")
                f.write("{}".format([x.to("cpu").detach().numpy() for x in outputs]) + "\n")
                f.write("{}".format([x.to("cpu").detach().numpy() for x in y]) + "\n")
                f.write("#################################################" + "\n")
                f.close()
                print("Loss at epoch {}, iteration {} is {}".format(epoch,
                                                                    i,
                                                                    running_loss / i) + "\n")
                print("{}".format([x.to("cpu").detach().numpy() for x in outputs]) + "\n")
                print("{}".format([x.to("cpu").detach().numpy() for x in y]) + "\n")
                print("#################################################" + "\n")


            if i % 2000 == 1999:  # print every 100 mini-batches
                f = open(output_file, "a")
                f.write("Checkpointing model" + "\n")
                torch.save(model.state_dict(), state_dict_file)
                f.close()
                print("Checkpointing model" + "\n")

        f = open(output_file, "a")
        f.write("###################################" + "\n")
        f.write("Loss for epoch {}: {}".format(epoch, running_loss) + "\n")
        f.write(str(outputs) + "\n")
        f.write(str(y) + "\n")
        f.write(str(loss) + "\n")
        running_loss = 0.0
        f.close()

        print("Checkpointing model" + "\n")
        torch.save(model.state_dict(), state_dict_file)
