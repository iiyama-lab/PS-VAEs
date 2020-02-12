import time
from options.train_options import TrainOptions
from data.data_loader import DataLoader
from models.combogan_model import ComboGANModel
from util.visualizer import Visualizer
import random
import csv
import numpy as np

opt = TrainOptions().parse()
dataset = DataLoader(opt)
testdata = DataLoader(opt, True)

print('# training images = %d' % len(dataset))

def train():
    model = ComboGANModel(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    # Update initially if continuing
    if opt.which_epoch > 0:
        model.update_hyperparams(opt.which_epoch)

    for epoch in range(opt.which_epoch + 1, opt.niter + opt.niter_decay + 1):
        print("")
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data, epoch)
            model.optimize_parameters()
        visualizer.display_current_results(model.get_current_visuals(), epoch)

        precision = 0
        if opt.est_mnist:
            for i, data in enumerate(testdata):
                model.set_input(data,epoch,val=True)
                precision += model.test(val=True)
            precision /= i + 1

        if not model.get_nanflag():
            print("kl loss becomes nan")
            return

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t, precision)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save(epoch)

        print('End of epoch %d / %d \t acc: %.3f \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, precision, time.time() - epoch_start_time))

        model.update_hyperparams(epoch)
train()
