import numpy as np


        
def build_data(n, nx, ny, channels =1, n_class = 2):


    images = np.zeros((n, nx, ny, channels))
    noised_images = np.zeros((n, nx, ny, channels))
    masks = np.zeros((n, nx, ny, channels))

    for i in range(0, n):
        image, label = create_image_and_label(ny, nx)
        
        labels = np.zeros((ny, nx, n_class), dtype=np.float32)
        labels[..., 1] = label
        labels[..., 0] = ~label
        
        labels = np.argmax(labels, axis=-1)
        labels = np.expand_dims(labels, axis=-1)
        images[i] = image
        noised_images[i] = noiseImg(image)
        masks[i] = labels

    return images, noised_images, masks


def create_image_and_label(ny, nx, cnt = 10, r_min = 5, r_max = 50, border = 92, sigma = 20):
    image = np.ones((nx, ny, 1))
    label = np.zeros((nx, ny, 2), dtype=np.bool)
    mask = np.zeros((nx, ny), dtype=np.bool)
    
    for _ in range(cnt):
        a = np.random.randint(border, nx - border)
        b = np.random.randint(border, ny - border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1,255)

        y, x = np.ogrid[-a : nx - a, -b : ny - b]
        m = x*x + y*y <= r*r
        mask = np.logical_or(mask, m)

        image[m] = h
        

    label[mask, 1] = 1
    
    return image, label[..., 1]


def noiseImg(image):
    
    """
    Args:
        image : numpy array of image        
    Return :
        noise_img : numpy array of image with gaussian noise added
    """
    
    mean=0
    random_var = np.random.uniform(0,127)
    
    gaus_noise = np.random.normal(mean, random_var , image.shape)

    noise_img = np.clip(image + gaus_noise, 0., 255.)
    return noise_img


