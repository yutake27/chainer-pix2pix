import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
from pathlib import Path

def out_image(updater, enc, dec, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = enc.xp

        w_in = 512
        w_out = 512
        in_ch = 3
        out_ch = 3

        in_all = np.zeros((n_images, in_ch, w_in, w_in)).astype("i")
        gt_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        gen_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")

        for it in range(n_images):
            batch = updater.get_iterator('test').next()
            batchsize = len(batch)

            x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
            t_out = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")

            for i in range(batchsize):
                x_in[i,:] = xp.asarray(batch[i][0])
                t_out[i,:] = xp.asarray(batch[i][1])
            x_in = Variable(x_in)

            z = enc(x_in)
            x_out = dec(z)

            in_all[it,:] = x_in.data.get()[0,:]
            gt_all[it,:] = t_out.get()[0,:]
            gen_all[it,:] = x_out.data.get()[0,:]

        def save_image(x, name, mode=None):
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            if C==1:
                x = x.reshape((rows*H, cols*W))
            else:
                x = x.reshape((rows*H, cols*W, C))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image_{}_{:0>8}.png'.format(name, trainer.updater.iteration)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x, mode=mode).convert('RGB').save(preview_path)

        x = np.asarray(np.clip(gen_all * 128 + 128, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gen")

        x = np.ones((n_images, 3, w_in, w_in)).astype(np.uint8)*255
        x[:,0,:,:] = 0
        for i in range(3):
            x[:,0,:,:] += np.uint8(15*i*in_all[:,i,:,:])
        save_image(x, "in", mode='HSV')

        x = np.asarray(np.clip(gt_all * 128+128, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gt")

    return make_image


def generate_image_from_contour(contour_path, enc, dec, out_dir_path):
    label = Image.open(contour_path)
    label = label.convert(mode='RGB')
    w_in = 512
    label = label.resize((w_in, w_in), Image.NEAREST)
    label = np.asarray(label).astype('f').transpose(2,0,1)/128.0-1.0

    xp = enc.xp
    in_ch = 3

    x_in = xp.zeros((1, in_ch, w_in, w_in)).astype('f')
    x_in[0,:] = xp.asarray(label)
    x_in = Variable(x_in)

    z = enc(x_in)
    x_out = dec(z)

    x = x_out.data[0,:]
    x = np.asarray(np.clip(x*128+128, 0.0, 255.0), dtype=np.uint8)
    x = x.transpose(1,2,0)

    out_dir_path = Path(out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    filename = out_dir_path/Path(contour_path).name

    print('generate: {}'.format(filename))
    Image.fromarray(x).convert(mode='RGB').save(filename)