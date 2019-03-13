from attnGan.attn_gan import AttnGAN


class VanillaSelfAttn(AttnGAN):

    def __init__(self, device, output_dir, opts, ixtoword, train_loader, val_loader):
        super().__init__(device, output_dir, opts, ixtoword, train_loader, val_loader)

    def build_models(self):
        pass

    def build_models_for_test(self, model_path):
        pass

    def train(self):
        pass

    def validate(self, netG, netsD, text_encoder, image_encoder):
        pass
