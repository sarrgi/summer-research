import cv2
import bovw_diff
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

def sift_draw(images, layers, sig, contrast, edge):
    kp_sum = 0
    total = 0

    #
    sift = cv2.SIFT.create(nOctaveLayers=layers, sigma=sig, edgeThreshold=edge, contrastThreshold=contrast)
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
            file_name = "".join((str(total), ".jpg"))
            print(file_name)
            cv2.imwrite(file_name, d)

            # track lengths
            kp_sum += len(kp)
            total += 1

    return kp_sum/total




# def drawKeyPts(im,keyp,col,th):
#     for curKey in keyp:
#         x=np.int(curKey.pt[0])
#         y=np.int(curKey.pt[1])
#         size = np.int(curKey.size)
#         cv2.circle(im,(x,y),size, col,thickness=th, lineType=8, shift=0)
#     plt.imshow(im)
#     return im

imWithCircles = drawKeyPts(origIm.copy(),keypoints,(0,255,0),5)


if __name__ == "__main__":

    train = bovw_diff.load_images_from_folder('/vol/grid-solar/sgeusers/sargisfinl/data/sorted_by_class_shell_0_reduced/train')  # take all images category by category
    test = bovw_diff.load_images_from_folder("/vol/grid-solar/sgeusers/sargisfinl/data/sorted_by_class_shell_0_reduced/test") # take test images

    output_file = open("out.txt", "w")

    kp_arr = []
    ax = []

    # for i in range(20, 51):
    layers = 3
    sig =  1.3
    contrast =  0.065
    edge = 1.8
    s = sift_draw(test, layers, sig, contrast, edge)    # print("layers:", i, ". sigma:", j/100, "- KP:", x)
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
