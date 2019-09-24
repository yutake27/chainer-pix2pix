import cv2
from pathlib import Path
import argparse

threshold = 240

def get_Contour_img():
    parser = argparse.ArgumentParser(description='get contour')
    parser.add_argument('--dir', '-d', type=str, help='input image dir')
    args = parser.parse_args()
    img_dir = Path(args.dir)
    out_dir = img_dir.parent/'Image_Contour'
    out_dir.mkdir(exist_ok=True)
    for img_path in img_dir.iterdir():
        out_path = out_dir/img_path.name
        if not out_path.exists():
            print(out_path)
            img = cv2.imread(str(img_path), 0)
            ret, img_thre = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            img_thre = cv2.Canny(img_thre, 200, 300, apertureSize=3)
            cv2.imwrite(str(out_path), img_thre)

if __name__ == '__main__':
    get_Contour_img()