import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pipeline import LandmarksPipeline

def main(pipeline, imagename):

    img = cv2.imread(imagename, 1)
    prepr, get_orig_coords = pipeline.preprocess(img)
    cv2.imwrite("test_preprocessed.jpg", prepr)

    image = np.array(Image.open('test_preprocessed.jpg'))

    for row in range(len(image)):
        for col in range(len(image[0])):
            image[row][col] = (image[row][col][0] - 102.9801,
                    image[row][col][1] - 115.9465,
                    image[row][col][2] - 122.7717)

    im_input = image[np.newaxis, np.newaxis, :, :][0].transpose((0, 3, 1, 2))
    pts, pts_orig = pipeline.forward_pass(im_input)
    pts_final = [get_orig_coords(pt) for pt in pts]
    pts_final = [(int(x), int(y)) for x, y in pts_final]
    pts_orig_final = [get_orig_coords(pt) for pt in pts_orig]
    pts_orig_final = [(int(x), int(y)) for x, y in pts_orig_final]
    for pt in pts_final:
        cv2.circle(img, pt, 5, (0, 255, 0), -1)
    img_copy = img.copy()
    for pt in pts_orig_final:
        cv2.circle(img_copy, pt, 5, (255, 0, 0), -1)


    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    img_copy = cv2.resize(img_copy, (0,0), fx=0.5, fy=0.5) 
    cv2.imshow('yeh', img)
    cv2.waitKey(0)
    cv2.imshow('yeh', img_copy)
    cv2.waitKey(0)
    print(pts_final)

if __name__ == "__main__":
    pipeline = LandmarksPipeline("lower")
    for img in os.listdir("./images"):
        imagename = "./images/{}".format(img)
        main(pipeline, imagename)
    cv2.destroyAllWindows()
