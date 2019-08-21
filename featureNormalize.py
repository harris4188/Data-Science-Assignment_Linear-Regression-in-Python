import numpy as np

data = open('ex1data2.txt').read().splitlines()
size = []
numberOfBedrooms = []

for x in range(len(data)):
    data[x] = data[x].split(',')
    size.append(int(data[x][0]))
    numberOfBedrooms.append(int(data[x][1]))

print(size)
print(numberOfBedrooms)

size = np.asarray(size)
numberOfBedrooms = np.asarray(numberOfBedrooms)

mean_size = np.mean(size)
std_size = np.std(size, ddof=1)

mean_numberOfBedrooms = np.mean(numberOfBedrooms)
std_numberOfBedrooms = np.std(numberOfBedrooms, ddof=1)

normalized_size = (size - mean_size)/std_size
normalized_numberOfBedrooms = (numberOfBedrooms - mean_numberOfBedrooms)/std_numberOfBedrooms

print(normalized_size)
print(normalized_numberOfBedrooms)

with open('normalized_features.txt', 'w') as f:
    f.write(str(mean_size))
    f.write('\n')
    f.write(str(std_size))
    f.write('\n')
    for x in range(len(normalized_size)):
        f.write(str(normalized_size[x]))
        if x < len(normalized_size) - 1:
            f.write(' ')
    f.write('\n')
    f.write(str(mean_numberOfBedrooms))
    f.write('\n')
    f.write(str(std_numberOfBedrooms))
    f.write('\n')
    for x in range(len(normalized_numberOfBedrooms)):
        f.write(str(normalized_numberOfBedrooms[x]))
        if x < len(normalized_numberOfBedrooms) - 1:
            f.write(' ')
