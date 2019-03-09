import torch

from damsm.damsm import Damsm
from networks import CNN_ENCODER, BERT_ENCODER


class DamsmBert(Damsm):

    def __init__(self, opts, update_interval, device):
        super().__init__(opts, update_interval, device)
        print("Bert Model")	

    def build_models(self, dict_size, batch_size, model_file_name):
        # build model ############################################################
        text_encoder = BERT_ENCODER(nhidden=self.opts.TEXT.EMBEDDING_DIM)
        image_encoder = CNN_ENCODER(self.opts.TEXT.EMBEDDING_DIM)

        text_encoder, image_encoder, start_epoch, optimizer = self.load_saved_model(text_encoder, image_encoder,
                                                                                    model_file_name)

        text_encoder = text_encoder.to(self.device)
        image_encoder = image_encoder.to(self.device)

        return text_encoder, image_encoder, start_epoch, optimizer

    def text_enc_forward(self, text_encoder, captions, input_mask):
        return text_encoder(captions, input_mask)
