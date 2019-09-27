from Util import load_mnist
import tensorflow as tf
from ops import *
import time
class AGAN(object):
    model_name='AGAN'
    def __init__(self,sess,epoch,batch_size,z_dim,checkpoint_dir,result_dir,log_dir):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir

        '''read two datasets'''
        self.input_height = 28
        self.input_width = 28
        self.output_height = 28
        self.output_width = 28
        self.c_dim = 1

        # train
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # test
        self.sample_num = 64  # number of generated images to be saved

        # load mnist
        self.data_X = load_mnist('./data/mnist_clothes/train-images-idx3-ubyte.gz')
        self.data_Y = load_mnist('./data/mnist_handwritting/train-images-idx3-ubyte.gz')
        # get number of batches for a single epoch
        self.num_batches = len(self.data_X) // self.batch_size
        return
    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            # 定义卷积操作，conv2d是作者封装的操作。采用
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            out_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)
            return out, out_logit, net

    def fitter(self,x, is_training=True, reuse=False):
        with tf.variable_scope("fitter", reuse=reuse):
            # 定义卷积操作，conv2d是作者封装的操作。采用
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='f_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='f_conv2'), is_training=is_training, scope='f_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 28*28*1, scope='f_fc3'), is_training=is_training, scope='f_bn3'))
            net = tf.reshape(net, [self.batch_size, 28 ,28 , 1])
            return net


        # with tf.variable_scope("generator", reuse=reuse):
        #     x = tf.reshape(x,[self.batch_size, 28*28*1])
        #     net = tf.nn.relu(bn(linear(x, 1024, scope='f_fc1'), is_training=is_training, scope='f_bn1'))
        #     net = tf.nn.relu(bn(linear(net, 28*28*1, scope='f_fc2'), is_training=is_training, scope='f_bn2'))
        #     net = tf.reshape(net, [self.batch_size, 28 ,28 , 1])
        #     return net

    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))
            return out
    def build_modle(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # the Y samples
        self.inputsY = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images_Y')
        self.inputsX = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images_X')
        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        """ Loss Function """

        # output of D for imagesY
        D_real_Y, D_real_logits_Y, _ = self.discriminator(self.inputsY, is_training=True, reuse=False)

        #output of D for imagesX
        F = self.fitter(self.inputsX, is_training=True, reuse=False)
        D_real_X, D_real_logits_X, _ = self.discriminator(F,is_training=True, reuse=True)

        # output of D for fake images
        G = self.generator(self.z, is_training=True, reuse=False)
        GF = self.fitter(G, is_training=True, reuse=True)
        D_fake_Z, D_fake_logits_Z, _ = self.discriminator(GF, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real_Y = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits_Y, labels=tf.ones_like(D_real_Y)))
        d_loss_real_X = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits_X, labels=tf.zeros_like(D_real_X)))
        d_loss_fake_Z = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_Z, labels=tf.zeros_like(D_fake_Z)))

        self.d_loss = 2*d_loss_real_Y + d_loss_real_X+d_loss_fake_Z

        # get loss for generator and fittor
        f_loss_real_X = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits_X, labels=tf.ones_like(D_real_X)))
        f_loss_real_Z = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_Z, labels=tf.zeros_like(D_fake_Z)))
        self.f_loss = f_loss_real_X + f_loss_real_Z

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_Z, labels=tf.ones_like(D_fake_Z)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()  # 返回需要训练的变量列表
        d_vars = [var for var in t_vars if 'd_' in var.name]
        f_vars = [var for var in t_vars if 'f_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.f_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.f_loss, var_list=[f_vars])
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 10, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=[g_vars])

        """" Testing """
        # for test
        self.fake_images_X = self.generator(self.z, is_training=False, reuse=True)
        self.fake_images_Y = self.fitter(self.fake_images_X, is_training=False, reuse=True)
        """ Summary 主要用于显示标量信息，tensorboard的一个函数，tensorboard用于参数可视化"""
        # d_loss_real_Y_sum = tf.summary.scalar("d_loss_real_Y", d_loss_real_Y)
        # d_loss_real_X_sum = tf.summary.scalar("d_loss_real_X", d_loss_real_X)
        # d_loss_fake_Z_sum = tf.summary.scalar("d_loss_fake_Z", d_loss_fake_Z)
        # d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        # g_loss_sum = tf.summary.scalar("gf_loss", self.gf_loss)
        #
        # # final summary operations
        # self.g_sum = tf.summary.merge([d_loss_fake_Z_sum, g_loss_sum])
        # self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        return
    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        d_loss=0
        f_loass=0
        g_loass=0
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            start_batch_id=0
            for idx in range(start_batch_id, self.num_batches):
                batch_images_X = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]#真实图像
                batch_images_Y = self.data_Y[idx * self.batch_size:(idx + 1) * self.batch_size]  # 真实图像
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)#均匀分布的噪声数据



                # self.writer.add_summary(summary_str, counter)
                if epoch<5:
                    # update F network
                    _, f_loss = self.sess.run([self.f_optim, self.f_loss],
                                                feed_dict={self.inputsX:batch_images_X,
                                                           self.z: batch_z})
                    # update G network
                    _, g_loss = self.sess.run([self.g_optim, self.g_loss],
                                          feed_dict={self.z: batch_z})
                else:
                    #update D network
                    _, d_loss = self.sess.run([self.d_optim, self.d_loss],
                                          feed_dict={self.inputsY: batch_images_Y,
                                                     self.inputsX: batch_images_X,
                                                     self.z: batch_z})
                    # update F network
                    _, f_loss = self.sess.run([self.f_optim, self.f_loss],
                                              feed_dict={self.inputsX: batch_images_X,
                                                         self.z: batch_z})
                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, f_loss: %.8f, g_loss:%.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, f_loss,g_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samplesX,samplesY = self.sess.run([self.fake_images_X,self.fake_images_Y],
                                            feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samplesX[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}_X.png'.format(
                                    epoch, idx))
                    save_images(samplesY[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(
                                    self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}_Y.png'.format(
                                    epoch, idx))
            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)


    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samplesX,samplesY = self.sess.run([self.fake_images_X,self.fake_images_Y],
                                          feed_dict={self.z: z_sample})

        save_images(samplesX[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_X.png')
        save_images(samplesY[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_Y.png')
    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #保存模型到第二个参数所指的文件中
        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


