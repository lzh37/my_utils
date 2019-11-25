import numpy as np

def mask2rle(img):
    # img : numpy array  1:mask, 0:background
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    
    
def rle2mask(rle):
    # rle : str
    # size：101 * 101
    mask = np.zeros((101,101))
    
    if not isinstance(rle,str):
        return mask
    
    str_list = rle.split(' ')
    mask = mask.reshape((-1,))
    for i in range(len(str_list)//2):
        mask[int(str_list[2*i])-1:int(str_list[2*i])+int(str_list[2*i+1])-1] = 1
    
    mask = mask.reshape((101,101)).T
    return mask