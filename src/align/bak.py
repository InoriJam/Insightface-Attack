#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:

import sys
import argparse
import tensorflow as tf
import tqdm
import numpy as np
import cv2
import os
import glob

from sklearn import metrics
from scipy import misc
from scipy.optimize import brentq
from scipy import interpolate

import lfw as lfw
import align.detect_face as FaceDet
import csv
import tensorflow.contrib.slim as slim
from models import inception_resnet_v1
import math
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
from losses.face_losses import arcface_loss


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

        bounding_boxes, points = FaceDet.detect_face(
            img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        num_face = bounding_boxes.shape[0]
        # assert num_face == 1, num_face
        if num_face != 1:
            bbox = [0, 0, 112, 112]
        else:
            bbox = bounding_boxes[0][:4]  # xy,xy

        margin = 32
        x0 = np.maximum(bbox[0] - margin // 2, 0)
        y0 = np.maximum(bbox[1] - margin // 2, 0)
        x1 = np.minimum(bbox[2] + margin // 2, img.shape[1])
        y1 = np.minimum(bbox[3] + margin // 2, img.shape[0])
        x0, y0, x1, y1 = bbox = [int(k + 0.5) for k in [x0, y0, x1, y1]]
        cropped = img[y0:y1, x0:x1, :]
        scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
        return scaled, bbox, points


def load_images():
    with open('securityAI_round1_dev.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join("./securityAI_round1_images/", row['ImageName'])
            raw_image = misc.imread(filepath)
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            yield row['ImageName'], raw_image, row["PersonName"]


def distance_to_victim(sess, img, victim_embeddings):
    emb = sess.run(any_embeddings, feed_dict={resized_input: [img]})
    dist = np.dot(emb, victim_embeddings.T).flatten()
    stats = np.percentile(dist, [10, 30, 50, 70, 90])
    return stats


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
        default=os.path.expanduser('~/data/LFW/MTCNN_160'))
    parser.add_argument('--eps', type=int, default=16, help='maximum pixel perturbation')
    args = parser.parse_args()

    '''input = tf.placeholder(tf.float32, shape=[None, 112, 112, 3], name='input')
    victim_embeddings_holder = tf.placeholder(tf.float32, shape=[None, 512], name='victim')
    resized_input = tf.to_float(tf.image.resize_images(input, (160, 160)))
    preprocessed_resized_input = (resized_input - 127.5) / 128.0
    prelogits, _ = inception_resnet_v1.inference(preprocessed_resized_input, 1.0, False, bottleneck_layer_size=512)
    any_embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

    #distance_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(victim_embeddings_holder, any_embeddings)), 1)) # maximize
    dot = tf.reduce_sum(tf.multiply(victim_embeddings_holder, any_embeddings), axis=1)
    norm = tf.linalg.norm(victim_embeddings_holder, axis=1) * tf.linalg.norm(any_embeddings, axis=1)
    distance_loss = tf.reduce_mean(tf.acos(dot / norm) / math.pi)
    grad_eval = tf.gradients(distance_loss, input)

    # insightface
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    preprocessed_input = (input - 127.5) * 0.0078125
    ins_face = get_resnet(preprocessed_input, 50, type='ir', w_init=w_init_method, trainable=False, keep_rate=1.0)
    ins_any_embeddings = tf.nn.l2_normalize(ins_face.outputs, 1, 1e-10, name='ins_embeddings')
    ins_dot = tf.reduce_sum(tf.multiply(victim_embeddings_holder, ins_any_embeddings), axis=1)
    ins_norm = tf.linalg.norm(victim_embeddings_holder, axis=1) * tf.linalg.norm(ins_any_embeddings, axis=1)
    ins_distance_loss = tf.reduce_mean(tf.acos(ins_dot / ins_norm) / math.pi)
    #ins_distance_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(victim_embeddings_holder, ins_any_embeddings)), 1))  # maximize
    ins_grad_eval = tf.gradients(ins_distance_loss, input)'''

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

    # total loss
    '''total_distance_loss = (ins_distance_loss + distance_loss) / 2
    total_grad_eval = tf.gradients(total_distance_loss, input)'''
    L2_sum = 0

    with tf.Session() as sess:
        '''variables = slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables if v.name.split('/')[0] == 'resnet_v1_50']
        saver2 = tf.train.Saver(variables_to_restore)
        saver2.restore(sess, './model/InsightFace_iter_best_710000.ckpt')
        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV1'))
        saver.restore(sess, './model/model-20180402-114759.ckpt-275')'''

        det = Detector()
        sess.run(tf.global_variables_initializer())
        for filename, img, name in load_images():

            imgfolder = os.path.join(args.data, name)
            assert os.path.isdir(imgfolder), imgfolder
            images = glob.glob(os.path.join(imgfolder, '*.png')) + glob.glob(os.path.join(imgfolder, '*.jpg'))
            images_list = [misc.imread(f) for f in images]
            resized_image_list = [misc.imresize(im, (160, 160)) for im in images_list]
            for im in resized_image_list:
                assert im.shape[0] == 160 and im.shape[1] == 160, \
                    "--data should only contain 160x160 images. Please read the README carefully."
            '''victim = sess.run(any_embeddings, feed_dict={resized_input: resized_image_list})

            ins_victim = sess.run(ins_face.outputs, feed_dict={input: images_list})
            print("Number of resized victim samples (the more the better): {}".format(len(victim)))'''
            R100_victim = sess.run(R100_any_embeddings, feed_dict={data_input: images_list})
            R50_victim = sess.run(R50_any_embeddings, feed_dict={data_input: images_list})
            R34_victim = sess.run(R34_any_embeddings, feed_dict={data_input: images_list})
            M_victim = sess.run(M_any_embeddings, feed_dict={data_input: images_list})

            print("attacking: " + filename)
            scaled_face, bbox, points = det.detect(img)

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

            if points.size:
                size = 13
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
                # grad, d_loss = sess.run((grad_eval, distance_loss), feed_dict={input: attack_img[np.newaxis, :], victim_embeddings_holder: victim})
                # ins_grad, ins_d_loss = sess.run((ins_grad_eval, ins_distance_loss), feed_dict={input: attack_img[np.newaxis, :], victim_embeddings_holder: ins_victim})
                total_grad, total_d_loss = sess.run((total_grad_eval, total_distance_loss),
                                                    feed_dict={data_input: attack_img[np.newaxis, :],
                                                               R100_victim_embeddings_holder: R100_victim,
                                                               R50_victim_embeddings_holder: R50_victim,
                                                               R34_victim_embeddings_holder: R34_victim,
                                                               M_victim_embeddings_holder: M_victim})
                # R100_grad, R100_d_loss = sess.run((R100_grad_eval, R100_distance_loss), feed_dict={R100_data: attack_img[np.newaxis, :], victim_embeddings_holder: R100_victim})
                # print(total_d_loss)
                n = np.sign(total_grad[0][0])
                n = np.multiply(n, template)
                attack_img = attack_img + (n * eps)
                attack_img = np.clip(attack_img, lower, upper)
                count += 1
                if total_d_loss > 2.2 or count == maxiter:
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

