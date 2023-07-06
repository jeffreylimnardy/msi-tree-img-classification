import numpy as np
import torch

data = np.load("assets/dataset.npy", allow_pickle=True)

Y = data[:, -1]

X = data[:, 0:-1]


def normalizeMatrix(inputMatrix):
    # eval evaluates the input string to a python list, then np.array converts it into a string
    inputMatrix = np.array(eval(inputMatrix))

    # normalize matrix from min to max range
    return (inputMatrix - np.min(inputMatrix)) / (np.max(inputMatrix) - np.min(inputMatrix))


# normalize the matrix to range 0 to 1
for i in range(np.size(X, 0)):
    for j in range(np.size(X, 1)):
        X[i][j] = normalizeMatrix(X[i][j])

# reshape the data cuz its weirdly structured af
numOfData = len(X)
pixelX = 5
pixelY = 5
channels = len(X[0])


# placeholder for 1st level (number of data points)
image = []
for m in range(numOfData):
    channel_matrix = []
    for n in range(channels):
        # placeholder for 2nd level (image channels)
        image_matrix = []
        for row in range(pixelX):
            # placeholder for 3rd level (image row)
            sub_list = []
            for col in range(pixelY):
                # placeholder for 4th level (image column)
                sub_list.append(X[m][n][row][col])
            image_matrix.append(sub_list)
        channel_matrix.append(image_matrix)
    image.append(channel_matrix)  # save restructured data in image matrix

# # placeholder for 1st level (number of data points)
# image = []
# for m in range(numOfData):
#     image_matrix = []
#     for row in range(pixelX):
#         # placeholder for 3rd level (image row)
#         sub_list = []
#         for col in range(pixelY):
#             # placeholder for 4th level (image column)
#             sub_list.append(list(X[m][n][row][col] for n in range(channels)))
#         image_matrix.append(sub_list)
#     image.append(image_matrix)  # save restructured data in image matrix


# convert image matrix to a numpy array and feed it into dataset
image = np.array(image)

np.save("assets/data_without_label.npy", image)

np.save("assets/labels.npy", Y)
