# -*- coding: UTF-8 -*-
from PIL import ImageGrab  # 截图、读取图片、保存图片
import numpy as np
import cv2
from paddlex import deploy
import pyautogui


class Controller(object):
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.threshold = 0.3
        self.x1, self.y1 = 735, 133
        self.x2, self.y2 = 1180 , 945
        self.cls_ = [
            'pt', 'yt', 'jz', 'nm',
            'mht', 'xhs', 'tz', 'bl', 'yz', 'xg', 'dxg'
        ]
        self.size = {
            'pt': 10, 'yt': 15,
            'jz': 20, 'nm': 25,
            'mht': 30, 'xhs': 35,
            'tz': 40, 'bl': 50,
            'yz': 55, 'xg': 60
        }
        self.init_model()
        self.count = 0

    def init_model(self):
        print('-[INFO] Loading...')
        self.det = deploy.Predictor('det_inference_model')
        print('-[INFO] Model loaded.')

    def draw(self, result, im):
        bboxes = []
        for value in result:
            xmin, ymin, w, h = np.array(value['bbox']).astype(np.int)
            xc = int(xmin + w / 2)
            yc = int(ymin + h / 2)
            cls_ = value['category']
            score = value['score']
            if score < self.threshold:
                continue
            color = (0, 255, 0)
            cv2.rectangle(im, (xmin, ymin), (xmin+w, ymin+h), color, 4)
            cv2.putText(im, '{:s}'.format(cls_),
                            (xmin, ymin), self.font, 1.0, (255, 0, 0), thickness=2)
            bboxes.append([self.x1 + xc, self.y1 + yc, cls_])
        return im, bboxes

    def infer(self, im, bboxes):
        YCs = [v[1] for v in bboxes]
        init_i = np.argmin(YCs)
        x, y, c = bboxes[init_i]
        print('-[INFO] {} in hand.'.format(c))
        bboxes.pop(init_i)
        same = [v for v in bboxes if v[2] == c]
        if len(same):
            S_YCs = [v[1] for v in same]
            S_XCs = [v[0] for v in same]
            tgt_i = np.argmin(S_YCs)
            x = S_XCs[tgt_i]
        pyautogui.click(x=x, y=y+100)
        print('-[Step]', x, y+100)

    def infer_v2(self, im, bboxes):
        bboxes = self.clip_bboxes(im, bboxes)
        YCs = [v[1] for v in bboxes]
        init_i = np.argmin(YCs)
        x, y, c = bboxes[init_i]
        print('-[INFO] {} in hand.'.format(c))
        bboxes.pop(init_i)
        same = [v for v in bboxes if v[2] == c]
        while not len(same) and self.cls_.index(c) < len(self.cls_) - 1:
            c = self.cls_[self.cls_.index(c) + 1]
            print('-[CHANGE] to', c)
            same = [v for v in bboxes if v[2] == c]
        if len(same):
            S_YCs = [v[1] for v in same]
            S_XCs = [v[0] for v in same]
            tgt_i = np.argmin(S_YCs)
            cv2.line(im, (x - self.x1, y - self.y1),
                             (S_XCs[tgt_i] - self.x1, S_YCs[tgt_i] - self.y1), 
                             (245, 0, 217), 2)
            x = S_XCs[tgt_i]
        else:
            x = np.random.randint(self.x1, self.x2)
        x += np.random.randint(-10, 10)
        pyautogui.click(x=x, y=y+100)
        print('-[Step]', x, y+100)

    def clip_bboxes(self, im, bboxes):
        YCs = [v[1] for v in bboxes]
        index = self.sorted_index(YCs)
        s_bboxes = [bboxes[i] for i in index]
        if len(s_bboxes):
            h_bboxes = [s_bboxes[0]]
        else:
            return []
        x_, y_, c_ = s_bboxes[0]
        s_bboxes.pop(0)
        if(len(s_bboxes) > 1):
            for x, y, c in s_bboxes:
                higher = [v[0] for v in s_bboxes if v[1] < y]
                count = 0
                for hx in higher:
                    if abs(x - hx) < 60:
                        count += 1
                if count > 1:
                    print('-[DROP] High ', x, y, c)
                    continue
                if self.cls_.index(c) >= self.cls_.index(c_):
                    biger = [
                        v[0] for v in s_bboxes if self.cls_.index(v[2]) > self.cls_.index(c) and v[1] < y]
                    for bx in biger:
                        if abs(bx - x) < 60:
                            print('-[DROP] Stack', x, y, c)
                            continue
                    h_bboxes.append([x, y, c])
                    cv2.line(im, (x - self.x1, y - self.y1),
                             (x_ - self.x1, y_ - self.y1), (245, 212, 217), 2)
                    cv2.circle(im, (x - self.x1, y - self.y1),
                               10, (0, 255, 0), -1)
        for x, y, c in h_bboxes:
            print('-[KEEP]', x, y, c)
        return h_bboxes

    def sorted_index(self, list_):
        return sorted(range(len(list_)), key=lambda k: list_[k])

    def run(self):
        self.count += 1
        image = ImageGrab.grab((self.x1, self.y1, self.x2, self.y2))
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        result = self.det.predict(image)
        image, bboxes = self.draw(result, image)
        for v in bboxes:
            print(v)
        if len(bboxes):
            self.infer_v2(image, bboxes)
        return image


if __name__ == '__main__':

    control = Controller()

    while True:
        image = control.run()
        cv2.imshow('image', image)
        cv2.waitKey(1200)
