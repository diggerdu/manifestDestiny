import time
import torch.backends.cudnn as cudnn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import sys
import time

cudnn.benchmark = True

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
# visualizer = Visualizer(opt)


model.load_network(model.netG, 'G', "latest")


total_steps = 0

embarkTime = time.time()

for epoch in range(1, opt.niter + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        model.set_input(data)
        for i in range(1000):
            model.optimize_parameters()
            errors = model.get_current_errors()
            if i % 30 == 0:
                print(errors)
            t_in = time.time() - epoch_start_time
    t = time.time() - epoch_start_time
    if errors['G_LOSS'] < 1e-14:
        break
    print('epoch ', epoch, ', current error is ', errors, ' cost time is ', t)


    # SAVE MODEL
    if time.time() - embarkTime > 60 * 48:
        model.save("latest")
        embarkTime = time.time()




