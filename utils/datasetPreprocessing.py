import numpy as np
import torch

data = np.load("assets/dataset_30bands.npy", allow_pickle=True)

Y = data[:, -1]

X = data[:, 0:-1]

print(X[0][0])
# print(Y[0])
print(np.max(X))
print(np.min(X))


def eval_strMatrix(inputMatrix):
    # eval evaluates the input string to a python list, then np.array converts it into a string
    inputMatrix = np.array(eval(inputMatrix))
    return inputMatrix


for i in range(np.size(X, 0)):
    for j in range(np.size(X, 1)):
        X[i][j] = eval_strMatrix(X[i][j])

print(X[0][0])
print("-" * 20)

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

# for i in range(image.shape[1]):
#     image[:, i, :, :] = (image[:, i, :, :] - np.min(image[:, i, :, :])) / \
#         (np.max(image[:, i, :, :]) - np.min(image[:, i, :, :]))


print(image[0][0])

np.savez("assets/data_preprocessed_30bands.npz",
         image=image, labels=Y)
