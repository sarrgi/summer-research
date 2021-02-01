import cv2
import bovw_diff
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px

def sift_draw(images, layers, sig):
    kp_sum = 0
    total = 0

    sift = cv2.SIFT.create(nOctaveLayers=layers, sigma=sig)
    for key,value in images.items():
        for img in value:
            # detect keypoints
            kp, des = sift.detectAndCompute(img,None)
            d = cv2.drawKeypoints(img, kp, img)

            # display image for 1 second
            # cv2.imshow("".join(("Keypint count:", str(len(kp)))), d)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            # track lengths
            kp_sum += len(kp)
            total += 1

    return kp_sum/total


if __name__ == "__main__":

    train = bovw_diff.load_images_from_folder('/vol/grid-solar/sgeusers/sargisfinl/data/bovw_0_color_run/train')  # take all images category by category
    test = bovw_diff.load_images_from_folder("/vol/grid-solar/sgeusers/sargisfinl/data/bovw_0_color_run/test") # take test images

    output_file = open("out.txt", "w")

    kp_arr = []
    ax = []

    for i in range(3, 4): #50
        for j in range(1, 1000):
            s = sift_draw(test, i, j/100)
            # print("layers:", i, ". sigma:", j/100, "- KP:", x)

            kp_arr.append(s)
            ax.append(j)

            # output_file.write("".join(("layers:", str(i), ". sigma:", str(j/100), "- KP:", str(x))))

    output_file.close()


    x = np.asarray(kp_arr, dtype=np.float32)
    y = np.asarray(ax, dtype=np.float32)
    # print(x,y)
    #
    # print(type(x), y)
    #
    # # plt.plot(kp_arr, ax)
    fig = px.scatter(x=x, y=y)
    fig.show()
    # plt.savefig('file.png')

    # plt.show()
    #
    # def
    # sifts = bovw_diff.sift_features(train)
