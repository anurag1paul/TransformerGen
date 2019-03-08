import os
import time
import datetime
import dateutil.tz

from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn

from damsm.damsm import Damsm
from damsm.damsm_bert import DamsmBert
from data_loader import CubDataset
from data_preprocess import DataPreprocessor
from utils import get_opts, make_dir, save_checkpoint, EpochTracker

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = 'dataset/'
output_directory = "checkpoints"
epoch_file = "epoch.txt"
log_file = "logs.log"

UPDATE_INTERVAL = 5
opts = EasyDict(get_opts("config/damsm_bert_bird.yaml"))


def create_loader(opts):
    preprocessor = DataPreprocessor("cub", data_dir)
    ixtoword = preprocessor.get_idx_to_word()
    train_set = CubDataset(preprocessor, opts, mode='train')
    val_set = CubDataset(preprocessor, opts, mode='val')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=False, pin_memory=True)
    return train_loader, val_loader, ixtoword


if __name__ == "__main__":
    if opts.TEXT.ENCODER == 'bert':
        damsm_model = DamsmBert(opts, UPDATE_INTERVAL, device)
    else:
        damsm_model = Damsm(opts, UPDATE_INTERVAL, device)
    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    
    output_dir = '%s/%s' % \
        (output_directory, opts.CHECKPOINTS_DIR)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    make_dir(model_dir)
    make_dir(image_dir)
    
    epoch_file = os.path.join(output_dir, "output.txt")
    epoch_tracker = EpochTracker(epoch_file)
    loss_log = open(os.path.join(model_dir, 'loss_log.txt'), 'a+')
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = opts.TREE.BASE_SIZE * (2 ** (opts.TREE.BRANCH_NUM-1))
    batch_size = opts.TRAIN.BATCH_SIZE

    train_loader, val_loader, ixtoword = create_loader(opts)

    # Train ##############################################################
    model_file_name = os.path.join(model_dir, 'checkpoint-' + str(epoch_tracker.epoch) + '.pth.tar')
    text_encoder, image_encoder, start_epoch, optimizer = damsm_model.build_models(len(ixtoword), batch_size, model_file_name)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(start_epoch, opts.TRAIN.MAX_EPOCH):
            epoch_start_time = time.time()
            count, s_train_loss, w_train_loss = damsm_model.train(train_loader, image_encoder, text_encoder, optimizer, epoch,
                          ixtoword, image_dir, batch_size)
            print('-' * 89)
            if len(val_loader) > 0:
                s_loss, w_loss = damsm_model.evaluate(val_loader, image_encoder, text_encoder)
                print('| end epoch {:3d} | val loss '
                      '{:5.2f} {:5.2f} |'
                      .format(epoch, s_loss, w_loss))
            print('-' * 89)
            
            loss_log.write('e:{} st:{} wt:{} sv:{} wv:{}\n'.format(epoch, s_train_loss, 
                                                                   w_train_loss, s_loss, w_loss))
            loss_log.flush()
            
            if  optimizer.param_groups[0]['lr'] > opts.TRAIN.ENCODER_LR/10.:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.98

            if (epoch % opts.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == opts.TRAIN.MAX_EPOCH):
                save_checkpoint({
                    'opts': opts,
                    'epoch': epoch,
                    'text_encoder': text_encoder,
                    'image_encoder': image_encoder,
                    'optimizer': optimizer
                }, epoch, model_dir)
                print('Saved models.')
            epoch_tracker.write(epoch)
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
