from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer import datasets

from net import Discriminator
from net import Encoder
from net import Decoder
from updater import FacadeUpdater

from facade_dataset import FacadeDataset
from facade_visualizer import out_image

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='../../Image',
                        help='Directory of image files.')
    parser.add_argument('--dataset_contour', '-c', default='../../Image_Contour',
                        help='Directory of contour image files')
    parser.add_argument('--data_num', '-n', type=int, default=400,
                        help='number of data to use')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=20,
                        help='Interval epoch of snapshot')
    parser.add_argument('--display_interval', type=int, default=1,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# snapshot interval: {}'.format(args.snapshot_interval))
    print('')

    # Set up a neural network to train
    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=3)
    dis = Discriminator(in_ch=3, out_ch=3)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    opt_dis = make_optimizer(dis)

    train_d, test_d = datasets.split_dataset_random(FacadeDataset(args.dataset, args.dataset_contour, args.data_num), args.data_num-25)
    #train_iter = chainer.iterators.MultiprocessIterator(train_d, args.batchsize, n_processes=14)
    #test_iter = chainer.iterators.MultiprocessIterator(test_d, args.batchsize, n_processes=14)
    train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize, shuffle=False)

    # Set up a trainer
    updater = FacadeUpdater(
        models=(enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec,
            'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    snapshot_interval = (args.snapshot_interval, 'epoch')
    display_interval = (args.display_interval, 'epoch')
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz'),
                   trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dec, 'dec_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(
    #     dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss', 'dec/loss', 'dis/loss'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(['enc/loss', 'dec/loss', 'dis/loss'], 'epoch', file_name='loss.png'),
                    trigger=display_interval)
    trainer.extend(
        out_image(
            updater, enc, dec,
            5, 5, args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
