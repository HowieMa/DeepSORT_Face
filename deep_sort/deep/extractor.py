from facenet_pytorch import InceptionResnetV1
import cv2
import torch
import numpy as np


class Extractor(object):
    def __init__(self, img_size=160, use_cuda=True):
        self.net = InceptionResnetV1(pretrained='vggface2')
        self.img_size = img_size    # 160

        self.use_cuda = use_cuda
        if use_cuda:
            self.net = self.net.cuda()

    def get_features(self, im_crops):
        # ori_img, cv2 array
        faces_im = []
        for face in im_crops:
            face = cv2.resize(face, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)  # (h, w, 3)
            face = face.transpose(2, 0, 1)                  # (3, H, W)
            face = torch.from_numpy(np.float32(face))
            face = (face - 127.5) / 128.0
            faces_im.append(face)
        faces_im = torch.stack(faces_im)

        return faces_im

    def __call__(self, im_crops):
        self.net.eval()

        img_batch = self.get_features(im_crops)

        with torch.no_grad():

            if self.use_cuda:
                img_batch = img_batch.cuda()
            embedding = self.net(img_batch)

        return embedding.cpu().numpy()
