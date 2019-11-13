import numpy as np
from scipy import ndimage as ndi

def find_energy(img, mask):
    h, w, c = img.shape
    y = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    xgrad = ndi.convolve1d(y, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(y, np.array([1, 0, -1]), axis=0, mode='wrap')
    ygrad[0] = y[1] - y[0]
    ygrad[h - 1] = y[h - 1] - y[h - 2]
    xgrad[:, 0] = y[:, 1] - y[:, 0]
    xgrad[:, w - 1] = y[:, w - 1] - y[:, w - 2]
    energy = np.sqrt(ygrad ** 2 + xgrad ** 2)
    if not mask is None:
        energy += h * w * 256 * mask
    return energy
    
def refind_energy(im):
    h, w = im.shape
    energy = im
    m = np.zeros((h, w))
    m[0] = im[0]
    
    for i in range(1, h):
        U = m[i-1]
        L = np.roll(U, 1)
        R = np.roll(U, -1)
        
        ULR = np.array([U, L, R])

        argmins = np.argmin(ULR, axis=0)
        tmp = np.choose(argmins, ULR)
        m[i] = im[i] + tmp
        m[i][0] = energy[i][0] + min(m[i-1][0], m[i-1][1])
        m[i][-1] = energy[i][-1] + min(m[i-1][-1], m[i-1][-2])

    return m

def del_seam(img, mask, energy):
    h, w = energy.shape
    carve_mask = np.ones((h, w), dtype=np.bool)
    coord = np.argmin(energy[-1])
    for i in range(h - 1, -1, -1):
        carve_mask[i][coord] = False
        tmp = coord
        if i != 0:
            if coord != 0 and energy[i - 1][coord - 1] <= energy[i - 1][coord]:
                tmp = coord - 1
            if coord != w - 1 and energy[i - 1][tmp] > energy[i - 1][coord + 1]:
                tmp = coord + 1
            coord = tmp
    #new = np.stack([carve_mask] * 3, axis=2)
    new_img = img[carve_mask].reshape((h, w - 1, 3))
    new_mask = mask[carve_mask].reshape((h, w - 1))
    
    return (new_img, new_mask, np.invert(carve_mask))

def add_seam(img, mask, energy):
    h, w = img.shape[0], img.shape[1]
    new_img = np.zeros((h, w + 1, 3))
    new_mask = np.zeros((h, w + 1))
    h, w = energy.shape
    carve_mask = np.zeros((h, w))
    coord = np.argmin(energy[-1])
    for i in range(h - 1, -1, -1):
        carve_mask[i][coord] = 1
        tmp = coord
        if tmp != w - 1:
            new = np.array([(img[i][tmp] + img[i][tmp + 1]) / 2])
            new_img[i, :, :] = np.concatenate((img[i, :(tmp + 1), :], new, img[i, (tmp + 1):, :]), axis=0)
            new = np.array([mask[i][tmp]])
            new_mask[i, :] = np.concatenate((mask[i, :(tmp + 1)], new, mask[i, (tmp + 1):]), axis=0)
        else:
            new = np.array([img[i][tmp]])
            new_img[i, :, :] = np.concatenate((img[i, :(tmp + 1), :], new), axis=0)
            new = np.array([mask[i][tmp]])
            new_mask[i, :] = np.concatenate((mask[i, :(tmp + 1)], new), axis=0)
        if i != 0:
            if coord != 0 and energy[i - 1][coord - 1] <= energy[i - 1][coord]:
                tmp = coord - 1
            if coord != w - 1 and energy[i - 1][tmp] > energy[i - 1][coord + 1]:
                tmp = coord + 1
            coord = tmp
    
    return (new_img, new_mask, carve_mask)

def seam_carve(img, mode, mask=None):
    
    is_none = mask is None
    if mode == "vertical shrink" or mode == "vertical expand":
        img = img.transpose(1, 0, 2)
        if not is_none:
            mask = mask.transpose()
    energy = find_energy(img, mask)
    energy = refind_energy(energy)
    
    if is_none:
        mask = np.zeros((img.shape[0], img.shape[1]))
    
    if mode == "vertical shrink" or mode == "horizontal shrink":
        new_img, new_mask, seam_mask = del_seam(img, mask, energy)
    else:
        new_img, new_mask, seam_mask = add_seam(img, mask, energy)
        
    if mode == "vertical shrink" or mode == "vertical expand":
        new_img = new_img.transpose(1, 0, 2)
        new_mask = new_mask.transpose()
        seam_mask = seam_mask.transpose()
    
    if is_none:
        new_mask = None
        
    return (new_img, new_mask, seam_mask)