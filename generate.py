import argparse
import os

import chainer
from chainer import training

from net import Discriminator
from net import Encoder
from net import Decoder
from updater import FacadeUpdater

from facade_dataset import FacadeDataset
from facade_visualizer import generate_image_from_contour

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--model', '-m', default='',
                        help='model snapshot')
    parser.add_argument('--enc', '-e', type=str, default='enc_iter_60000.npz', help='encoder snapshot')
    parser.add_argument('--dec', '-d', type=str, default='dec_iter_60000.npz', help='decoder snapshot')
    parser.add_argument('--out', '-o', type=str, default='out', help='output dir')
    parser.add_argument('--input', '-i', default='sample.jpg', help='input jpg', required=True)
    parser.add_argument('--contour', '-c', action='store_true', help='from contour image or not')
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

    if args.model:
        opt_enc = make_optimizer(enc)
        opt_dec = make_optimizer(dec)
        opt_dis = make_optimizer(dis)

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
    elif args.enc and args.dec:
        chainer.serializers.load_npz(args.enc, enc)
        chainer.serializers.load_npz(args.dec, dec)

    if not args.contour:
        from make_contour import get_contour_image
        get_contour_image(args.input)

    generate_image_from_contour(args.input, enc, dec, args.out)

if __name__ == '__main__':
    main()