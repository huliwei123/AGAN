import os
from agan import AGAN
import Util as util
import tensorflow as tf
import argparse
import pylab as pl
import numpy as np
"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())#进行命令解析

"""checking arguments,to make sure they are not bad arguments"""
def check_args(args):
    # --checkpoint_dir
    util.check_folder(args.checkpoint_dir)

    # --result_dir
    util.check_folder(args.result_dir)

    # --result_dir
    util.check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def main():
    args = parse_args()
    if args is None:
      exit()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        asso_gan = AGAN(sess,args.epoch,args.batch_size,args.z_dim,
                            args.checkpoint_dir,args.result_dir,args.log_dir)
        if asso_gan is None:
            raise Exception("[!]create assocoation_gan failed")

        # build graph,according to the gan_type you input in args
        asso_gan.build_modle()

        # show network architecture
        # show_all_variables()

        # launch the graph in a session
        asso_gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        # gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")

def testmain():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        asso_gan = AGAN(sess,20,64,62,
                            'checkpoint','result','log')
        if asso_gan is None:
            raise Exception("[!]create assocoation_gan failed")

        # build graph,according to the gan_type you input in args
        asso_gan.build_modle()
        # show network architecture
        # show_all_variables()

        # launch the graph in a session
        asso_gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        # gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")


def testPlt():
    times = np.arange(0,9)  # times为x的值，0为起点，5为终点，0,01为步长
    fun=[9,8,7,6,5,4,3,2,1]
    pl.plot(times, fun)  # 画图
    pl.xlabel("epoch")  # x轴的标记
    pl.ylabel("err")  # y轴的标记
    pl.title("no title")  # 图的标题
    pl.show()  # 显示图


if __name__ == '__main__':
    testmain()
