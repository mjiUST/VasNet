import time
import torch
from options import TrainOptions
from dataset import dataset_unpair
from model import UID
from saver import Saver
from data import CreateDataLoader
from util.visualizer import Visualizer

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  # visualizer
  visualizer = Visualizer(opts)

  # data loader
  print('\n--- load dataset ---')
  # dataset = dataset_unpair(opts)    
  # train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
  data_loader = CreateDataLoader(opts)

  # model
  print('\n--- load model ---')
  model = UID(opts)
  model.setgpu(opts.gpu)
  if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_it = 0
  else:
    ep0, total_it = model.resume(opts.resume)
  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)

  # train
  print('\n--- train ---')
  max_it = 500000
  for ep in range(ep0, opts.n_ep):
    print("Epoch: {}".format(ep))

    for it, data in enumerate(data_loader):
      images_a, images_b = data['A'], data['B']
      if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
        continue
      images_a = images_a.cuda(opts.gpu).detach()
      images_b = images_b.cuda(opts.gpu).detach()

      # update model
      # model.update_D_content(images_a, images_b)  # uncomment for GAN_content, discriminator of z_content
      model.update_D(images_a, images_b)
      if (it + 1) % 2 != 0 and it != len(data_loader)-1:
        continue
      model.update_EG()

      losses_dic = model.get_current_losses()
      visualizer.plot_current_losses(ep, float(it)/len(data_loader), opts, losses_dic)

      # save to display file
      if (it+1) % 48 == 0:
        print('total_it: %d (ep %d, it %d), lr %08f' % (total_it+1, ep, it+1, model.gen_opt.param_groups[0]['lr']))
        print('Dis_I_loss: %04f, Dis_B_loss %04f, GAN_loss_I %04f, GAN_loss_B %04f' % (model.disA_loss, model.disB_loss, model.gan_loss_i,model.gan_loss_b))
        print('B_percp_loss %04f, Recon_II_loss %04f' % (model.B_percp_loss, model.l1_recon_II_loss))
      if (it+1) % 200 == 0:
        saver.write_img(ep*len(data_loader) + (it+1), model)
        
      total_it += 1
      if total_it >= max_it:
        saver.write_img(-1, model)
        saver.write_model(-1, model)
        break

    # decay learning rate
    if opts.n_ep_decay > -1:
      model.update_lr()

    saver.write_img(ep, model)
    # Save network weights
    saver.write_model(ep, total_it+1, model)

  return

if __name__ == '__main__':
  main()
