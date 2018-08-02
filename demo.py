import argparse
from pathlib import Path

import cv2

from darkflow.net.build import TFNet

options = {'model': 'cfg/yolo-1c.cfg', 'load': 17500, 'threshold': 0.3, 'labels': 'modd2-labels.txt'}


def main(demo_dataset_dir, n_samples):
    tfnet = TFNet(options)

    images = list(Path(demo_dataset_dir).glob('*.jpg'))

    if n_samples == -1:
        n_samples = len(images)

    for img in images[0:n_samples]:

        imgcv = cv2.imread(str(img))
        result = tfnet.return_predict(imgcv)
        for res in result:
            cv2.rectangle(imgcv,
                          (res['topleft']['x'], res['topleft']['y']),
                          (res['bottomright']['x'],
                           res['bottomright']['y']),
                          (0, 255, 0), 4)
            text_x, text_y = res['topleft'][
                                 'x'] - 10, res['topleft']['y'] - 10

            cv2.putText(imgcv, res['label'], (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


        cv2.imshow('image', imgcv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_dataset_dir', type=str, help='Dataset to be used for demo')
    parser.add_argument('--n_samples', type=int, default=-1, help='Dataset to be used for demo')

    args = parser.parse_args()
    main(args.demo_dataset_dir, args.n_samples)
