import cv2
import bovw_diff
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px

def sift_draw(images, layers, sig, edge, contrast):
    kp_sum = 0
    total = 0

    #
    sift = cv2.SIFT.create(nOctaveLayers=3, sigma=2.5, contrastThreshold=0.065, edgeThreshold=3)
    for key,value in images.items():
        for img in value:
            # detect keypoints
            kp, des = sift.detectAndCompute(img,None)
            d = cv2.drawKeypoints(img, kp, img)

            # display image for 1 second
            # cv2.imshow("".join(("Keypint count:", str(len(kp)))), d)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            # save image
            # file_name = "".join(("images/sift/edge_", str(edge/10), "/", key, "_", str(total), ".jpg"))
            # print(file_name)
            # cv2.imwrite(file_name, d)

            # track lengths
            kp_sum += len(kp)
            total += 1

    return kp_sum/total


if __name__ == "__main__":

    train = bovw_diff.load_images_from_folder('/vol/grid-solar/sgeusers/sargisfinl/data/sorted_by_class_shell_0_reduced/train')  # take all images category by category
    test = bovw_diff.load_images_from_folder("/vol/grid-solar/sgeusers/sargisfinl/data/sorted_by_class_shell_0_reduced/test") # take test images

    output_file = open("out.txt", "w")

    kp_arr = []
    ax = []

    # for i in range(20, 51):
    s = sift_draw(test, 1, 1, 1, 0.065)    # print("layers:", i, ". sigma:", j/100, "- KP:", x)
    print(s)
    #
    # # kp_arr.append(s)
    # # ax.append(j)
    #
    # # output_file.write("".join(("layers:", str(i), ". sigma:", str(j/100), "- KP:", str(x))))
    #
    # output_file.close()


    # x = np.asarray(kp_arr, dtype=np.float32)
    # y = np.asarray(ax, dtype=np.float32)
    #
    # fig = px.scatter(x=x, y=y)
    # fig.show()
