'''
This code includes all FundusAug modules.
This code is partially borrowed from https://github.com/HzFu/EyeQ_Enhancement
'''
import math
import numpy as np
import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
'''import joblib
from joblib import Parallel, delayed'''

types=['brightness','contrast','saturation','hue','sharpness','HALO','HOLE','SPOT','BLUR']

levels = {  'SH': [0,2],
            'HS': [[0,2],[1, 1.2, 1.5, 2],[0.8,0.1,0.05,0.05]],
            'HG': [[0.4,0.9],[-0.06,-0.01]],
            'SP': [[1,8],[0.01,0.08]],
            'IB': [0.1,3]
        }

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
            
        return image, mask

class Sharpness(object):
    def __init__(self, level=None, prob = 0.5):
        
        self.level_range = levels['SH']
        self.prob = prob

    def __get_parameter(self):
        return np.random.uniform(self.level_range[0], self.level_range[1])

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            #print('Sharpness')
            sharpness_factor = self.__get_parameter()
            image = F.adjust_sharpness(image, sharpness_factor)
        return image, target

class Halo(object):
    def __init__(self, size, brightness_factor=None, prob = 0.5):
        self.size = size
        self.zeros = torch.zeros((self.size, self.size)).cuda()
        
        self.brightness_range = levels['HS'][0]
        self.degrad_weight = levels['HS'][1]
        self.degrad_weight_prob = levels['HS'][2]
        
        self.brightness_factor = brightness_factor
        self.prob = prob

    def __get_parameter(self):
        # -> brightness_factor, de_w
        
        brightness_factor = np.random.uniform(self.brightness_range[0],self.brightness_range[1])  
        de_w = np.random.choice(self.degrad_weight,1, p=self.degrad_weight_prob)
        de_w = de_w[0]
            
        return brightness_factor, de_w

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            #print('Halo')
            
            brightness_factor, weight_hal = self.__get_parameter()
            # weight_r = [251/255,141/255,177/255]
            # weight_g = [249/255,238/255,195/255]
            # weight_b = [246/255,238/255,147/255]

            r=random.randint(80,130)
            g=random.randint(200,240)
            b=random.randint(145,175)
            weight_r = [251/255, 141/255, 177/255, r/255]
            weight_g = [249/255, 238/255, 195/255, g/255]
            weight_b = [246/255, 238/255, 147/255, b/255]

            if brightness_factor >= 0.2:
                num=random.randint(1,3)
            else:
                num=random.randint(0,3)

            w0_a = random.randint(self.size/2-int(self.size/8),self.size/2+int(self.size/8))
            center_a = [w0_a, w0_a]
            wei_dia_a = 0.75 + (1.0-0.75) * random.random()
            dia_a = self.size*wei_dia_a
            Y_a, X_a = np.ogrid[:self.size, :self.size]
            
            dist_from_center_a = np.sqrt((X_a - center_a[0]) ** 2 + (Y_a - center_a[1]) ** 2)
            circle_a = dist_from_center_a <= (int(dia_a / 2))

            mask_a = self.zeros.clone()
            mask_a[circle_a] = torch.mean(image) #np.multiply(A[0], (1 - t))

            center_b = center_a
            Y_b, X_b = np.ogrid[:self.size, :self.size]
            dist_from_center_b = np.sqrt((X_b - center_b[0]) ** 2 + (Y_b - center_b[1]) ** 2)

            dia_b_max =2* int(np.sqrt(max(center_a[0],self.size-center_a[0])*max(center_a[0],self.size-center_a[0])+max(center_a[1],self.size-center_a[1])*max(center_a[1],self.size-center_a[1])))/self.size
            wei_dia_b = 1.0+(dia_b_max-1.0) * random.random()

            if num ==0:
                # if halo tend to be a white one, set the circle with a larger radius.
                dia_b = self.size * wei_dia_b + abs(max(center_b[0] - self.size / 2, center_b[1] - self.size / 2) + self.size*2 / 3)
            else:
                dia_b = self.size * wei_dia_b + abs(max(center_b[0]-self.size/2,center_b[1]-self.size/2)+self.size/2)

            circle_b = dist_from_center_b <= (int(dia_b / 2))

            mask_b = self.zeros.clone()
            mask_b[circle_b] = torch.mean(image)

            # weight_hal0 = [0, 1, 1.5, 2, 2.5]
            # delta_circle = torch.abs(mask_a - mask_b) * weight_hal0[1]
            delta_circle = torch.abs(mask_a - mask_b) * weight_hal
            
            dia = max(center_a[0],self.size-center_a[0],center_a[1],self.size-center_a[1])*2
            gauss_rad = int(np.abs(dia-dia_a))
            sigma = 2/3*gauss_rad+0.01

            if(gauss_rad % 2) == 0:
                gauss_rad= gauss_rad+1
            delta_circle = F.gaussian_blur(torch.reshape(delta_circle,(1,self.size,self.size)), (gauss_rad, gauss_rad), sigma)
            
            if num==0 or num==1 or num==2:
                delta_circle = torch.stack([(weight_r[num]*delta_circle),(weight_g[num]*delta_circle),(weight_b[num]*delta_circle)])
            else:
                num=1
                delta_circle = torch.stack([(weight_r[num]*delta_circle),(weight_g[num]*delta_circle),(weight_b[num]*delta_circle)])
            
            image = image + torch.reshape(delta_circle,(3, self.size, self.size))
            image = torch.clamp(image, min=0, max=1)
            
        return image, target
    
class Hole(object):
    def __init__(self,size,prob = 0.5):
        self.size = size
        self.prob = prob
        self.level_range = levels['HG']

        self.min_diameter_circle = self.level_range[0][0]
        self.max_diameter_circle = self.level_range[0][1]
        self.min_brightness_factor = self.level_range[1][0]
        self.max_brightness_factor = self.level_range[1][1]

        self.zeros = torch.zeros((self.size, self.size)).cuda()

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            #print('Hole')
            # diameter_circle = random.randint(int(0.4 * self.size), int(0.7 * self.size))
            diameter_circle = random.randint(int(self.min_diameter_circle * self.size), int(self.max_diameter_circle * self.size))
            center =[random.randint(self.size/4,self.size*3/4),random.randint(self.size*3/8,self.size*5/8)]
            Y, X = np.ogrid[:self.size, :self.size]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            circle = dist_from_center <= (int(diameter_circle/2))

            mask = self.zeros.clone()
            mask[circle] = 1

            num_valid = torch.sum(target)
            aver_color = torch.sum(image) / (3*num_valid)
            if aver_color>0.25:
                brightness = random.uniform(-0.26,-0.262)
                brightness_factor = random.uniform((brightness-0.06*aver_color), brightness-0.05*aver_color)
            else:
                brightness = 0
                brightness_factor = 0
            # print( (aver_color,brightness,brightness_factor))
            mask = mask * brightness_factor

            rad_w = random.randint(int(diameter_circle*0.55), int(diameter_circle*0.75))
            rad_h = random.randint(int(diameter_circle*0.55), int(diameter_circle*0.75))
            sigma = 2/3 * max(rad_h, rad_w)*1.2

            if (rad_w % 2) == 0: rad_w = rad_w + 1
            if(rad_h % 2) ==0 : rad_h =rad_h + 1

            mask = F.gaussian_blur(torch.reshape(mask,(1,self.size,self.size)), (rad_w, rad_h), sigma)
            mask = torch.stack([mask, mask, mask])

            image = image + torch.reshape(mask,(3,self.size,self.size))
            image = torch.clamp(image, min=0, max=1)

        return image, target

class Spot(object):
    def __init__(self,size,center=None, radius=None,prob = 0.5):
        self.size = size
        self.center = center
        self.radius = radius
        self.prob = prob
        self.zeros = torch.zeros((self.size, self.size)).cuda()
        self.ones = torch.ones((3,1)).cuda()
    def __call__(self, image, target=None):
        if random.random() < self.prob:
            #print('Spot')
            s_num =random.randint(5,10)
            mask0 = self.zeros.clone()
            center=self.center
            radius=self.radius
            for i in range(s_num):

                radius = random.randint(math.ceil(0.01*self.size),int(0.05*self.size))

                center  = [random.randint(radius+1,self.size-radius-1),random.randint(radius+1,self.size-radius-1)]
                Y, X = np.ogrid[:self.size, :self.size]
                dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
                circle = dist_from_center <= (int(radius/2))

                k = (14/25) +(1.0-radius/25)
                beta = 0.5 + (1.5 - 0.5) * (radius/25)
                A = k * self.ones.clone()
                d =0.3 *(radius/25)
                t = math.exp(-beta * d)

                mask = self.zeros.clone()
                mask[circle] = torch.multiply(A[0], torch.tensor(1-t).cuda())
                mask0 = mask0 + mask
                mask0[mask0 != 0] = 1

                sigma = (5 + (20 - 0) * (radius/25))*2
                rad_w = random.randint(int(sigma / 5), int(sigma / 4))
                rad_h = random.randint(int(sigma / 5), int(sigma / 4))
                if (rad_w % 2) == 0: rad_w = rad_w + 1
                if (rad_h % 2) == 0: rad_h = rad_h + 1

                mask = F.gaussian_blur(torch.reshape(mask,(1,self.size,self.size)), (rad_w, rad_h), sigma)
                mask = torch.stack([mask, mask, mask])
                
                image = image + torch.reshape(mask,(3,self.size,self.size))
                image = torch.clamp(image,min=0,max=1)

        return image, target
    
class Blur(object):
    def __init__(self, sigma=None, prob = 0.5):
        self.level_range = levels['IB']
        self.sigma = sigma
        self.prob = prob

    def __get_parameter(self):
        return np.random.uniform(self.level_range[0], self.level_range[1])

    def __call__(self, image, target=None):
        if random.random() < self.prob:
        
            if self.sigma is None:
                sigma = self.__get_parameter()
            else:
                sigma = self.sigma
            
            rad_w = random.randint(int(sigma/3), int(sigma/2))
            if (rad_w % 2) == 0: rad_w = rad_w + 1
            rad_h = rad_w
            image = F.gaussian_blur(image, (rad_w,rad_h), sigma)

        return image, target
