import argparse
import tensorflow as tf
import numpy as np
import os
import glob

from scipy import misc
import align.detect_face as FaceDet
import csv
import math

# 先将src目录加到PYTHONPATH
class Detector():
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = FaceDet.create_mtcnn(sess, None)

    def detect(self, img):
        """
        img: rgb 3 channel
        """
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        _, points = FaceDet.detect_face(
            img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        return points


def load_images():
    with open('securityAI_round1_dev.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join("./securityAI_round1_images/", row['ImageName'])
            raw_image = misc.imread(filepath)
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            yield row['ImageName'], raw_image, row["PersonName"]


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
        dist = np.mean(dist)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
        dist = np.mean(dist)
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', help='path to MTCNN-aligned LFW dataset',
        default=os.path.expanduser('./lfw/new112'))
    parser.add_argument('--eps', type=int, default=25, help='maximum pixel perturbation')
    args = parser.parse_args()

    # Resnet100
    data_input = tf.placeholder(tf.float32, shape=(None, 112, 112, 3), name='data_input')
    from model.resnet100 import tf_resnet100

    R100_data, R100_fc1 = tf_resnet100.KitModel('./model/resnet100/resnet100.npy', data_input)
    R100_victim_embeddings_holder = tf.placeholder(tf.float32, shape=[None, 512], name='R100_victim')
    R100_any_embeddings = tf.nn.l2_normalize(R100_fc1, 1, 1e-10, name='R100_embeddings')
    R100_distance_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(R100_victim_embeddings_holder, R100_any_embeddings)), 1))
    R100_grad_eval = tf.gradients(R100_distance_loss, R100_data)

    # Resnet50
    from model.resnet50 import tf_resnet50

    R50_data, R50_fc1 = tf_resnet50.KitModel('./model/resnet50/resnet50.npy', data_input)
    R50_victim_embeddings_holder = tf.placeholder(tf.float32, shape=[None, 512], name='R50_victim')
    R50_any_embeddings = tf.nn.l2_normalize(R50_fc1, 1, 1e-10, name='R50_embeddings')
    R50_distance_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(R50_victim_embeddings_holder, R50_any_embeddings)), 1))
    R50_grad_eval = tf.gradients(R50_distance_loss, R50_data)

    # Resnet34
    from model.resnet34 import tf_resnet34

    R34_data, R34_fc1 = tf_resnet34.KitModel('./model/resnet34/resnet34.npy', data_input)
    R34_victim_embeddings_holder = tf.placeholder(tf.float32, shape=[None, 512], name='R34_victim')
    R34_any_embeddings = tf.nn.l2_normalize(R34_fc1, 1, 1e-10, name='R34_embeddings')
    R34_distance_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(R34_victim_embeddings_holder, R34_any_embeddings)), 1))
    R34_grad_eval = tf.gradients(R34_distance_loss, R34_data)

    # Mobilenet
    from model.mobilenet import tf_mobilenet

    M_data, M_fc1 = tf_mobilenet.KitModel('./model/mobilenet/mobilenet.npy', data_input)
    M_victim_embeddings_holder = tf.placeholder(tf.float32, shape=[None, 128], name='M_victim')
    M_any_embeddings = tf.nn.l2_normalize(M_fc1, 1, 1e-10, name='M_embeddings')
    M_distance_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(M_victim_embeddings_holder, M_any_embeddings)), 1))
    M_grad_eval = tf.gradients(M_distance_loss, M_data)

    # total_loss
    total_distance_loss = (R100_distance_loss + R50_distance_loss + R34_distance_loss + M_distance_loss) / 4.0
    total_grad_eval = tf.gradients(total_distance_loss, data_input)

    L2_sum = 0

    with tf.Session() as sess:

        det = Detector()
        sess.run(tf.global_variables_initializer())
        for filename, img, name in load_images():

            imgfolder = os.path.join(args.data, name)
            assert os.path.isdir(imgfolder), imgfolder
            images = glob.glob(os.path.join(imgfolder, '*.png')) + glob.glob(os.path.join(imgfolder, '*.jpg'))
            images_list = [misc.imread(f) for f in images]
            R100_victim = sess.run(R100_any_embeddings, feed_dict={data_input: images_list})
            R50_victim = sess.run(R50_any_embeddings, feed_dict={data_input: images_list})
            R34_victim = sess.run(R34_any_embeddings, feed_dict={data_input: images_list})
            M_victim = sess.run(M_any_embeddings, feed_dict={data_input: images_list})

            print("attacking: " + filename)
            points = det.detect(img)

            print("R100 Distance of ORIG: ",
                  distance(sess.run(R100_any_embeddings, feed_dict={data_input: [img]}), R100_victim,
                           distance_metric=0))
            print("R50 Distance of ORIG: ",
                  distance(sess.run(R50_any_embeddings, feed_dict={data_input: [img]}), R50_victim, distance_metric=0))
            print("R34 Distance of ORIG: ",
                  distance(sess.run(R34_any_embeddings, feed_dict={data_input: [img]}), R34_victim, distance_metric=0))
            print("M Distance of ORIG: ",
                  distance(sess.run(M_any_embeddings, feed_dict={data_input: [img]}), M_victim, distance_metric=0))
            '''print("Distance of ORIG: ", distance(sess.run(ins_any_embeddings,feed_dict={input:[img]}), ins_victim, distance_metric=1))
            print("Distance of resized ORIG: ", distance(sess.run(any_embeddings,feed_dict={input:[img]}), victim, distance_metric=1))'''

            attack_img = img.copy().astype(np.float)
            lower = np.clip(attack_img - args.eps, 0., 255.)
            upper = np.clip(attack_img + args.eps, 0., 255.)
            count = 0
            maxiter = 100
            eps = 1

            # 通过MTCNN侦测到人脸特征，然后对人脸关键部位进行扰动
            if points.size:
                size = 13 #参数忘记了，也许是15（笑）
                template = np.zeros_like(img)
                template[int(points[5][0]) - size:int(points[5][0]) + size,
                int(points[0][0]) - size:int(points[0][0]) + size] = 1
                template[int(points[6][0]) - size:int(points[6][0]) + size,
                int(points[1][0]) - size:int(points[1][0]) + size] = 1
                template[int(points[7][0]) - size:int(points[7][0]) + size,
                int(points[2][0]) - size:int(points[2][0]) + size] = 1
                template[int(points[8][0]) - size:int(points[8][0]) + size,
                int(points[3][0]) - size:int(points[3][0]) + size] = 1
                template[int(points[9][0]) - size:int(points[9][0]) + size,
                int(points[4][0]) - size:int(points[4][0]) + size] = 1
            else:
                template = np.ones_like(img)

            while True:
                total_grad, total_d_loss = sess.run((total_grad_eval, total_distance_loss),
                                                    feed_dict={data_input: attack_img[np.newaxis, :],
                                                               R100_victim_embeddings_holder: R100_victim,
                                                               R50_victim_embeddings_holder: R50_victim,
                                                               R34_victim_embeddings_holder: R34_victim,
                                                               M_victim_embeddings_holder: M_victim})
                n = np.sign(total_grad[0][0])
                n = np.multiply(n, template)
                attack_img = attack_img + (n * eps)
                attack_img = np.clip(attack_img, lower, upper)
                count += 1
                if total_d_loss > 2 or count == maxiter:
                    break
            print("R100 Distance of ADV: ",
                  distance(sess.run(R100_any_embeddings, feed_dict={data_input: [attack_img]}), R100_victim,
                           distance_metric=0))
            print("R50 Distance of ADV: ",
                  distance(sess.run(R50_any_embeddings, feed_dict={data_input: [attack_img]}), R50_victim,
                           distance_metric=0))
            print("R34 Distance of ADV: ",
                  distance(sess.run(R34_any_embeddings, feed_dict={data_input: [attack_img]}), R34_victim,
                           distance_metric=0))
            print("M Distance of ADV: ",
                  distance(sess.run(M_any_embeddings, feed_dict={data_input: [attack_img]}), M_victim,
                           distance_metric=0))
            diff = attack_img.reshape(-1, 3) - img.reshape(-1, 3)
            L2 = np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
            print(filename + ", L2: " + str(L2))
            L2_sum += L2
            misc.imsave("./images/" + filename, attack_img.astype(np.uint8))
        L2_aver = L2_sum / 712.0
        print("aver L2: " + str(L2_aver))

