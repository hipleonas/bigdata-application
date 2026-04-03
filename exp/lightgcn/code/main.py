import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import psutil
import os

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

total_start_time = time.time()


patience = 100
best_recall = 0.0
epochs_without_improvement = 0
test_interval = 20

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        if epoch % test_interval == 0:
            cprint(f"[TEST - Epoch {epoch}]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

            current_recall = results['recall'][0]

            if current_recall > best_recall:
                best_recall = current_recall
                epochs_without_improvement = 0

                torch.save(Recmodel.state_dict(), weight_file)
                print(f"[*] Store best model: {best_recall:.4f}")
            else:
                epochs_without_improvement += test_interval
                print(f"[!] The quality not improve. Early Stopping: {epochs_without_improvement}/{patience} epochs")

                if epochs_without_improvement >= patience:
                    print(f"[EARLY STOPPING] activate at epoch {epoch}. Stop the training early!")
                    break

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)

        #Compute resources
        end = time.time()
        epoch_time = end - start

        process = psutil.Process(os.getpid())
        ram_usage_mb = process.memory_info().rss / (1024 ** 2)
        cpu_usage = psutil.cpu_percent()

        gpu_mem_mb = 0
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information} | T:{epoch_time:.2f}s | RAM:{ram_usage_mb:.1f}MB | CPU:{cpu_usage}% | VRAM:{gpu_mem_mb:.1f}MB')

        if world.tensorboard:
            w.add_scalar('System_Metrics/Epoch_Time_s', epoch_time, epoch)
            w.add_scalar('System_Metrics/RAM_Usage_MB', ram_usage_mb, epoch)
            w.add_scalar('System_Metrics/CPU_Usage_Percent', cpu_usage, epoch)
            if torch.cuda.is_available():
                w.add_scalar('System_Metrics/GPU_Memory_MB', gpu_mem_mb, epoch)

finally:
    total_end_time = time.time()
    total_duration_mins = (total_end_time - total_start_time) / 60
    print(f"Total runtime: {total_duration_mins:.2f} minutes")
    print(f">> Model has the best Recall: {best_recall:.4f}")

    if world.tensorboard:
        w.close()