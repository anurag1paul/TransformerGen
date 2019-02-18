import os
import shutil
import socket
import time
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

import utils
from data_loader import CubDataset
from data_preprocess import DataPreprocessor

device = "cuda" if torch.cuda.is_available() else "cpu"

captions_path = '../cub_dataset/text_c10'
img_name_path = '../cub_dataset/images.txt'
data_path = '../cub_dataset/images/'
output_directory = "checkpoints/"
epoch_file = "epoch.txt"
log_file = "logs.log"


def create_loader(opts):
    preprocessor = DataPreprocessor("cub", img_name_path, data_path, captions_path)
    train_set = CubDataset(device, preprocessor, mode='train')
    val_set = CubDataset(device, preprocessor, mode='val')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)
    return train_loader, val_loader


def main():
    model = None
    opts = utils.get_opts()
    epoch_tracker = utils.EpochTracker(epoch_file, log_file)

    if epoch_tracker.epoch > 0:
        name = output_directory + epoch_tracker.epoch + ".pth"
        model.load_state_dict(torch.load(name))

    train_loader, val_loader = create_loader(opts)

    # different modules have different learning rate
    optimizer = torch.optim.SGD(model.get_params(), lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)

    # You can use DataParallel() whether you use Multi-GPUs or not
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # when training, use reduceLROnPlateau to reduce learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opts.lr_patience)

    # loss function
    criterion = None

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    best_txt = os.path.join(output_directory, 'best.txt')
    # create log
    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    start_epoch = epoch_tracker.epoch

    for epoch in range(start_epoch, opts.epochs):

        # remember change of the learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = float(param_group['lr'])
            logger.add_scalar('Lr/lr_' + str(i), old_lr, epoch)

        train(train_loader, model, criterion, optimizer, epoch, logger, device, opts)  # train for one epoch
        result, img_merge = validate(val_loader, model, epoch, logger, opts)  # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result.inception_score < best_result.inception_score
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write("epoch={}, IC:{}".format(epoch, result.inception_score))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        # save checkpoint for each epoch
        utils.save_checkpoint({
            'args': opts,
            'epoch': epoch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch, output_directory)

        epoch_tracker.write(epoch, 0)
        # when rml doesn't fall, reduce learning rate
        scheduler.step(result.absrel)

    logger.close()


# train
def train(train_loader, model, criterion, optimizer, epoch, logger, device, opts):
    model.train()  # switch to train mode
    end = time.time()

    batch_num = len(train_loader)

    for i, (input, target) in enumerate(train_loader):

        input = input.to(device)
        target = target.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()

        with torch.autograd.detect_anomaly():
            # pred = model(input)
            # loss = criterion(pred_ord, target_c)
            optimizer.zero_grad()
            loss.backward()  # compute gradient and do SGD step
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        gpu_time = time.time() - end

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# validation
def validate(val_loader, model, epoch, logger, opts):

    model.eval()  # switch to evaluate mode

    end = time.time()

    skip = len(val_loader) // 9  # save images every skip iters
    loss = 0
    img_sample = None

    for i, (input, target) in enumerate(val_loader):

        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred, _ = model(input)

        torch.cuda.synchronize()
        gpu_time = time.time() - end

    return loss, img_sample


if __name__ == '__main__':
    main()
