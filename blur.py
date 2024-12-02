import numpy as np

class ImageBlurrer:
    def __init__(self):
        pass
        
    def average_blur(self, image, kernel_size=5):
        """Average/mean blur with uniform kernel"""
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        if len(image.shape) == 3: 
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:,:,c] = self._apply_kernel(image[:,:,c], kernel)
            return result
        else:  # Grayscale
            return self._apply_kernel(image, kernel)
            
    def gaussian_blur(self, image, kernel_size=5, sigma=1):
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        kernel /= np.sum(kernel)
        
        if len(image.shape) == 3:  
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:,:,c] = self._apply_kernel(image[:,:,c], kernel)
            return result
        else: 
            return self._apply_kernel(image, kernel)
            
    def median_blur(self, image, kernel_size=3):
        if len(image.shape) == 3: 
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                padded = np.pad(image[:,:,c], 
                              ((kernel_size//2, kernel_size//2), 
                               (kernel_size//2, kernel_size//2)), 
                              mode='edge')
                filtered = np.zeros_like(image[:,:,c])
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
                        filtered[i, j] = np.median(neighborhood)
                result[:,:,c] = filtered
            return result
        else:  
            padded = np.pad(image, ((kernel_size//2, kernel_size//2), 
                                   (kernel_size//2, kernel_size//2)), 
                           mode='edge')
            filtered = np.zeros_like(image)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
                    filtered[i, j] = np.median(neighborhood)
            return filtered
            
    def _apply_kernel(self, image, kernel):
        """Helper method to apply convolution with a kernel"""
        kernel_size = kernel.shape[0]
        padded = np.pad(image, 
                       ((kernel_size//2, kernel_size//2), 
                        (kernel_size//2, kernel_size//2)), 
                       mode='edge')
        output = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)
        return output