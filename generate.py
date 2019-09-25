import argparse
import os

import chainer
from chainer import training

from net import Discriminator
from net import Encoder
from net import Decoder
from updater import FacadeUpdater

from facade_dataset import FacadeDataset
from facade_visualizer import out_image
import shutil

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--model', '-m', default='',
                        help='model snapshot')
    parser.add_argument('--input', '-i', default='sample.jpg',
                        help='input jpg')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))

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

    if os.path.exists('generate_tmp'):
        shutil.rmtree('generate_tmp')

    os.mkdir('generate_tmp')
    shutil.copyfile(args.input,'generate_tmp/tmp.jpg')
    test_d = FacadeDataset('generate_tmp/', 'generate_tmp/', 1)
    test_iter = chainer.iterators.SerialIterator(test_d, 1)


    # Set up a trainer
    updater = FacadeUpdater(
        models=(enc, dec, dis),
        iterator={},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec,
            'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (200, 'epoch'), out='generate/')
    chainer.serializers.load_npz(args.model, trainer)


    out_image(
        updater, enc, dec,
        1, 1, args.seed, 'generate/',True,test_iter)(trainer)


if __name__ == '__main__':
    main()