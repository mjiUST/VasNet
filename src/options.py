import argparse

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--preproc', type=str, default='None', help='preprocessing methods: None, suppress_half')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--resize_size_x', type=int, default=144, help='resized image size for training,144 for face')
    self.parser.add_argument('--resize_size_y', type=int, default=144, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=128, help='cropped image size for training, 128 for face')
    self.parser.add_argument('--input_dim_a', type=int, default=1, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=1, help='# of input channels for domain B')
    self.parser.add_argument('--nThreads', type=int, default=1, help='# of threads for data loader')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
    self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=1, help='freq (epoch) of saving models')

    # training related
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=80, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=40, help='epoch start decay learning rate, set -1 if no decay') 
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--gpu', type=str, default='cuda', help='gpu ids: e.g. 0  0,1,2, 0,2')
    self.parser.add_argument('--lambdaB', type=float, default=0.1, help='perceptual loss weight for B')
    self.parser.add_argument('--lambdaI', type=float, default=0, help='perceptual loss weight for I')
    self.parser.add_argument('--percp_layer', type=int, default=14, help='the layer of feature for perceptual loss')
    self.parser.add_argument('--percep', type=str, default='default', help='type of perceptual loss: default, face, multi')
    self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')

    ### dataloader (copy from Cycle_GAN)
    self.parser.add_argument('--isTrain', action='store_true')
    self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
    self.parser.add_argument('--disable_data_aug', action='store_true', help='if true, do not randomly crop / affine / flip')
    self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
    self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
    self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
    self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    self.parser.add_argument('--num_threads', type=int, default=6, help='# of threads for data loader')
    self.parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
    self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
    self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

    ### visualizer
    self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
    self.parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
    self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')


  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--preproc', type=str, default='None', help='preprocessing methods: None, suppress_half')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=128, help='cropped image size for training')
    self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
    self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    self.parser.add_argument('--input_dim_a', type=int, default=1, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=1, help='# of input channels for domain B')
    self.parser.add_argument('--a2b', type=int, default=1, help='translation direction, 1 for a2b, 0 for b2a')
    self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')

    # ouptput related
    self.parser.add_argument('--num', type=int, default=1, help='number of outputs per image')
    self.parser.add_argument('--name', type=str, default='test', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='./outputs', help='path for saving result images and models')
    self.parser.add_argument('--orig_dir', type=str, default='./outputs', help='path for saving result images and models')
    self.parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
    self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=1, help='freq (epoch) of saving models')

    # model related
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    self.parser.add_argument('--lambdaB', type=float, default=0.1, help='perceptual loss weight for B')
    self.parser.add_argument('--lambdaI', type=float, default=10, help='color loss weight')
    self.parser.add_argument('--percep', type=str, default='default', help='type of perceptual loss: default(vgg19), face(vggface), multi')
    self.parser.add_argument('--percp_layer', type=int, default=14, help='the layer of feature for perceptual loss')

    ### dataloader (copy from Cycle_GAN)
    self.parser.add_argument('--isTrain', action='store_true')
    self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
    self.parser.add_argument('--disable_data_aug', action='store_true', help='if true, do not randomly crop / affine / flip')
    self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
    self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
    self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
    self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    self.parser.add_argument('--num_threads', type=int, default=6, help='# of threads for data loader')
    self.parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
    self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
    self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    # set irrelevant options
    self.opt.dis_scale = 3
    self.opt.dis_norm = 'None'
    self.opt.dis_spectral_norm = False
    return self.opt
