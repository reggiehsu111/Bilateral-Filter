import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r, border_type='reflect'):
        
        self.border_type = border_type
        self.sigma_r = sigma_r
        self.set_sigma_s(sigma_s)
        self.set_sigma_r(sigma_r)


    def set_sigma_s(self, sigma_s):
        self.sigma_s = sigma_s
        self.range = 3*self.sigma_s
        self.w_size = 2*self.range+1
        self.Gs = self.spatial_kernel()

    def set_sigma_r(self, sigma_r):
        self.sigma_r = self.sigma_r



    def spatial_kernel(self):
        x, y = np.meshgrid(np.linspace(-self.range,self.range,self.w_size), np.linspace(-self.range,self.range,self.w_size))
        Gs = np.exp(-( (x*x+y*y) / ( 2 * self.sigma_s**2 ) ) )
        return Gs

    def range_kernel(self, guidance_region):
        mid_pixel = guidance_region[self.range, self.range]
        Gr = np.exp(-((guidance_region-mid_pixel).astype(np.float64)**2) / ( 2 * self.sigma_r**2 ))
        # If 3 channels
        if Gr.shape[-1] == 3:
            Gr = Gr[:,:,0]*Gr[:,:,1]*Gr[:,:,2]
        # Gr must have only 1 channel
        return Gr

    def spatial_kernel_1d(self):
        x = np.linspace(-self.range,self.range,self.w_size)
        Gs_1d = np.exp(-(x*x) / (2 * self.sigma_s**2))
        return Gs_1d


    def joint_bilateral_filter(self, input, guidance):
        
        self.input_shape = input.shape
        input = np.pad(input, ((self.range,self.range),(self.range,self.range),(0,0)), 'symmetric')
        self.norm_guidance = guidance/255

        if self.norm_guidance.shape[-1] ==3:
            self.norm_guidance = np.pad(self.norm_guidance, ((self.range,self.range),(self.range,self.range),(0,0)), 'symmetric')
        else:
            self.norm_guidance = np.pad(self.norm_guidance, (self.range,self.range), 'symmetric')


        output = np.zeros(self.input_shape)

        for x in range(self.input_shape[0]):
            for y in range(self.input_shape[1]):

                input_region = input[x:x+self.w_size, y:y+self.w_size]
                guidance_region = self.norm_guidance[x:x+self.w_size, y:y+self.w_size]
                self.Gr = self.range_kernel(guidance_region)
                self.F = self.Gr*self.Gs

                for i in range(3):
                    output[x,y,i] = np.sum(self.F*input_region[:,:,i]) / np.sum(self.F)
        return output


