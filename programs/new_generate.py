import pickle
import random
import cv2
import numpy as np

training_file = "../traffic-signs-data/train.p"
validation_file = "../traffic-signs-data/valid.p"
testing_file = "../traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

coords = train['coords']
features = train['features']
labels = train['labels']
sizes = train['sizes']

l0 = min(labels)
l1 = max(labels)
n_labels = l1 - l0 + 1

f_labels = np.zeros([n_labels], np.int32)


for value in labels:
    f_labels[value] += 1


# Sort images index according its label
f_idx = []

# Create a list to store them

for i in range(0, n_labels):
    f_idx.append([])

for idx, value in enumerate(labels):
    f_idx[value].append(idx)

new_images = []
new_labels = []


for idx, n in enumerate(f_labels):
    max_f = n * 3
    indexes = f_idx[idx]
    for i in range(n+1, max_f):
        # Select a random index between 0 and nmax
        pos = random.randint(0, len(indexes)-1)
        idx_img = indexes[pos]

        # Ara recuperem la imatge

        img = features[idx_img]
        label = labels[idx_img]

        # Will change luminance by a 50% factor up or down. Crazy

        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        channels = cv2.split(img)
        strength = random.uniform(0.5, 1.5)
        channels[0] = cv2.multiply(channels[0], strength)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        cols = img.shape[0]
        rows = img.shape[1]

        # Rotem +/- 20º

        angle = random.uniform(-20.0, 20.0)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

        # Fem un resize entre 18 i 32 pixels

        lsize = random.randint(18, 32)
        dst = cv2.resize(dst, (lsize, lsize))
        b0 = int((32 - lsize)/2)
        b1 = 32 - lsize - b0
        dst = cv2.copyMakeBorder(dst, b0, b1, b0, b1, cv2.BORDER_REPLICATE)

        # Desplacem entre 0 i 5 pixels em ambdues direccions
        dx = random.uniform(-5.0, 5.0)
        dy = random.uniform(-5.0, 5.0)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        dst = cv2.warpAffine(dst, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

        new_images.append(dst)
        new_labels.append(label)

new_images = np.array(new_images)
new_labels = np.array(new_labels)

features = np.concatenate((features, new_images), axis=0)
labels = np.concatenate((labels,  new_labels), axis=0)

dict = {'features': features, 'labels': labels}

with open("../traffic-signs-data/super_train.p", mode='wb') as f:
    pickle.dump(dict, f)

print(len(new_images))
