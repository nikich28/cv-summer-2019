import numpy as np
from skimage.transform import rescale
from skimage.transform import pyramid_gaussian
from copy import deepcopy

def cross_cor(im1, im2, i, j):
    h, w = im1.shape
    new_2 = np.roll(np.roll(im2, j, axis=0), i, axis=1)
    first = im1[max(0, j):min(h, h + j), max(0, i):min(w, w + i)]
    second = new_2[max(0, j):min(h, h + j), max(0, i):min(w, w + i)]
    return (first * second).sum() / np.sqrt((first ** 2).sum() * (second ** 2).sum())

def mse(im1, im2, i, j):
    h, w = im1.shape
    new_2 = np.roll(np.roll(im2, j, axis=0), i, axis=1)
    first = im1[max(0, j):min(h, h + j), max(0, i):min(w, w + i)]
    second = new_2[max(0, j):min(h, h + j), max(0, i):min(w, w + i)]
    return ((first - second) ** 2).sum() / first.size

def best_match_mse(img1, img2, d1, d2, r_y, r_x):
    best = 1000000
    best_x, best_y = 0, 0
    for i in range(r_y - d1, r_y + d2 + 1):
        for j in range(r_x - d1, r_x + d2 + 1):
            k = mse(img1, img2, i, j)
            if k < best:
                best = k
                best_x, best_y = i, j
    return (best_x, best_y)

def best_match_cor(img1, img2, d1, d2, r_y, r_x):
    best = 0
    best_x, best_y = 0, 0
    for i in range(r_y - d1, r_y + d2 + 1):
        for j in range(r_x - d1, r_x + d2 + 1):
            k = cross_cor(img1, img2, i, j)
            if k >= best:
                best = k
                best_x, best_y = i, j
    return (best_x, best_y)

def bgr_division(img):
    ch = []
    h, w = img.shape[0] // 3, img.shape[1]
    dy = h // 20
    dx = w // 20
    ch.append(img[dy:h-dy, dx:-dx])
    ch.append(img[h + dy:2 * h - dy, dx:-dx])
    ch.append(img[2*h + dy:3 * h - dy, dx:-dx])
    return ch

def pyr(bgr):
    a, b = bgr[0].shape
    if a < 500 and b < 500:
        b_eps = best_match_mse(bgr[1], bgr[0], 9, 9, 0, 0)
        r_eps = best_match_mse(bgr[1], bgr[2], 9, 9, 0, 0)
        return (b_eps, r_eps)
    else:
        pyr_b = pyramid(bgr[0])
        pyr_g = pyramid(bgr[1])
        pyr_r = pyramid(bgr[2])
        b_eps = best_match_mse(pyr_g[-1], pyr_b[-1], 9, 9, 0, 0)
        r_eps = best_match_mse(pyr_g[-1], pyr_r[-1], 9, 9, 0, 0)
        for i in range(len(pyr_b) - 2, 0, -1):
            b_eps = tuple(map(lambda x: x * 2, b_eps))
            r_eps = tuple(map(lambda x: x * 2, r_eps))
            b_eps = best_match_mse(pyr_g[i], pyr_b[i], 0, 1, b_eps[0], b_eps[1])
            r_eps = best_match_mse(pyr_g[i], pyr_r[i], 0, 1, r_eps[0], r_eps[1])
    return (tuple(map(lambda x: x * 2, b_eps)), tuple(map(lambda x: x * 2, r_eps)))

def pyramid(img):
    p = [img]
    while img.shape[0] >= 500 or img.shape[1] >= 500:
        new = np.zeros((img.shape[0] // 2 + img.shape[0] % 2, img.shape[1] // 2 + img.shape[1] % 2))
        for i in range(new.shape[0]):
            new[i] = img[i * 2, ::2]
        p.append(new)
        img = new
    return p
    
def align(img, g_coords):
    height = img.shape[0] // 3
    bgr = bgr_division(img)
    b_eps, r_eps = pyr(bgr)
    x_b, y_b = b_eps
    x_r, y_r = r_eps
    img = np.dstack((np.roll(np.roll(bgr[2], x_r, axis=1), y_r, axis=0), bgr[1], np.roll(np.roll(bgr[0], x_b, axis=1), y_b, axis=0)))
    return img, (g_coords[0] - y_b - height, g_coords[1] - x_b), (g_coords[0] - y_r + height, g_coords[1] - x_r)
