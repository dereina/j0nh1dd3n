import os
from os import walk
from os.path import join
import glob
import argparse
import cv2
from cv2 import FLOODFILL_FIXED_RANGE
from matplotlib.pyplot import polar
import numpy as np

from numba import jit

@jit(nopython=True)
def sphericalCoordinates(pixel):
        x = pixel[0]
        y = pixel[1]
        z = pixel[2]     
        polarAngle = np.degrees(np.arctan2(y,x))
        azimuthAngle = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z))
        radialDistance = np.sqrt(x**2 + y**2 + z**2)
        return polarAngle, azimuthAngle, radialDistance

@jit(nopython=True)
def cartesianCoordinates(pixel):
    polarAngle, azimuthAngle, radialDistance = pixel
    polarAngle = np.radians(polarAngle)
    azimuthAngle = np.radians(azimuthAngle)
    x = radialDistance * np.cos(polarAngle) * np.sin(azimuthAngle)
    y = radialDistance * np.sin(polarAngle) * np.sin(azimuthAngle)
    z = radialDistance * np.cos(azimuthAngle)
    return x, y, z

@jit(nopython=True)
def aggregateMasksImage(image, ponderation, mask):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            sphericals = sphericalCoordinates(image[y][x])
            coord_x = int(np.round(sphericals[0]))
            coord_y = int(np.round(sphericals[1]))
            mask[coord_y][coord_x] = cartesianCoordinates((sphericals[0], sphericals[1], 255))#image[y][x] 
            ponderation[coord_y][coord_x] += 1/(1 + ponderation[coord_y][coord_x]**2)

@jit()
def aggregateMasks(file, ponderation, mask):
    image = cv2.imread(file)
    aggregateMasksImage(image, ponderation, mask)

@jit(parallel=True)
def fullColorMask(output):
    mask = np.zeros((90, 90, 3)).astype(np.float)
    i=0
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            mask[y][x] = cartesianCoordinates((x, y, 255));
    
    cv2.imwrite(output+"/full_color_mask.png", mask.astype(np.uint8))
    return mask

@jit()
def bezierContrast(mask):
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            t = mask[y][x]
            t =  (1-t)**5 * 0 + 5*t*(1-t)**4 * 0 + 10*t**2*(1-t)**3 * 0 + 10*t**3*(1-t)**2 * 1 + 5*t**4*(1-t) * 1 + t**5 * 1
            mask[y][x] =  (1-t)**5 * 0 + 5*t*(1-t)**4 * 0 + 10*t**2*(1-t)**3 * 0 + 10*t**3*(1-t)**2 * 1 + 5*t**4*(1-t) * 1 + t**5 * 1
            #if mask[y][x] < 0.25:
            #    mask[y][x] = 0

            #else:
            #    mask[y][x] = 1

@jit(nopython=True)
def imageDivide(mask, ponderation):
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            mask[y][x] /= ponderation[y][x]

@jit()
def maskFromFiles(image_files):
    mask = np.zeros((90, 90, 3)).astype(np.float)
    ponderation = np.zeros((90, 90)).astype(np.float)
    i=0
    for file in image_files:
        aggregateMasks(file, ponderation, mask) 

    #mask = mask / ponderation
    #imageDivide(mask, ponderation)
    #for y in range(mask.shape[0]):
    #    for x in range(mask.shape[1]):
    #        mask[y][x] /= ponderation[y][x]

    #ponderation = ponderation/np.max(ponderation)

    return mask, ponderation

@jit(nopython=True)
def filterImage(image, mask):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            sphericals = sphericalCoordinates(image[y][x])
            coord_x = int(np.round(sphericals[0]))
            coord_y = int(np.round(sphericals[1]))
            image[y][x] =  image[y][x] * mask[coord_y][coord_x]

@jit(nopython=True)
def countPixels(image, threshold):
    count = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x] > threshold:
                count += 1
            
    return count

@jit()
def segmentImages(output, image_files, mask):
    for file in image_files:
        image = cv2.imread(file)
        filterImage(image ,mask)
        
        name = file.split("\\")[-1]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        count =countPixels(image, 64)
        # Creating kernel
        kernel = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((5, 5), np.uint8)
        
        # Using cv2.erode() method 
        image = cv2.erode(image, kernel) 
        image = cv2.dilate(image, kernel2) 
        image = cv2.dilate(image, kernel2) 

        cv2.imwrite(output+"/"+str(count)+name, image)


class Segmentation():
    def __init__(self, foreground_path, background_path, inference_path, mask_path, output, mask_threshold, pixel_count):
        self.foreground_path = foreground_path
        self.background_path = background_path
        self.inference_path = inference_path
        self.mask_path = mask_path
        self.output = output
        self.mask_threshold = mask_threshold
        self.pixel_count = pixel_count
        self.foreground_images = []
        self.background_images = []
        self.inference_images = []
        try:
            os.mkdir(self.output, 0x755 );
           
        except:
            pass

    def loadSegmentationImages(self):
        for file in glob.glob(self.foreground_path+"/*.png"):
            print(file)
            #dict_images_X[dir_name].append(file)
            self.foreground_images.append(file)

        for file in glob.glob(self.background_path+"/*.png"):
            print(file)
            #dict_images_X[dir_name].append(file)
            self.background_images.append(file)    

    def loadinferenceImages(self):
        for file in glob.glob(self.inference_path+"/*.png"):
            print(file)
            #dict_images_X[dir_name].append(file)
            self.inference_images.append(file)

         

    def sphericalCoordinates(self, pixel):
        x = pixel[0]
        y = pixel[1]
        z = pixel[2]     
        polarAngle = np.degrees(np.arctan2(y,x))
        azimuthAngle = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z))
        radialDistance = np.sqrt(x**2 + y**2 + z**2)
        return polarAngle, azimuthAngle, radialDistance
    
    def cartesianCoordinates(self, pixel):
        polarAngle, azimuthAngle, radialDistance = pixel
        polarAngle = np.radians(polarAngle)
        azimuthAngle = np.radians(azimuthAngle)
        x = radialDistance * np.cos(polarAngle) * np.sin(azimuthAngle)
        y = radialDistance * np.sin(polarAngle) * np.sin(azimuthAngle)
        z = radialDistance * np.cos(azimuthAngle)
        return x, y, z

    def aggregateMasks(self, file, ponderation, mask):
        image = cv2.imread(file)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                sphericals = self.sphericalCoordinates(image[y][x])
                coord_x = int(np.round(sphericals[0]))
                coord_y = int(np.round(sphericals[1]))
                mask[coord_y][coord_x] += image[y][x] 
                ponderation[coord_y][coord_x] += 1/(1 + ponderation[coord_y][coord_x])
                #if mask[coord_y][coord_x] == 255:
                #    mask[coord_y][coord_x] -=1  
                
    def fullColorMask(self):
        mask = np.zeros((90, 90, 3)).astype(np.float)
        i=0
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                mask[y][x] = self.cartesianCoordinates((x, y, 255));
       
        cv2.imwrite(self.output+"/full_color_mask.png", mask.astype(np.uint8))
        return mask

    def maskFromFiles(self, image_files):
        mask = np.zeros((90, 90, 3)).astype(np.float)
        ponderation = np.zeros((90, 90)).astype(np.float)
        i=0
        for file in image_files:
            self.aggregateMasks(file, ponderation, mask) 

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                mask[y][x] /= ponderation[y][x]

        #ponderation = ponderation/np.max(ponderation)
    
        return mask, ponderation

    def computeMask(self):
        maskf, ponderationf = maskFromFiles(self.foreground_images)
    
        maskb, ponderationb = maskFromFiles(self.background_images)

        #ponderationf /= np.max(ponderationf)
        #ponderationb /= np.max(ponderationb)
        #ponderation_segm /= np.max(ponderation_segm)
        
        #ponderation_segm = (ponderationf/(1+ponderationb))
        ponderation_segm = (-ponderationb+ponderationf).clip(min=0)

        ponderationf = ponderationf/np.max(ponderationf)
        ponderationb = ponderationb/np.max(ponderationb)
        #ponderation_segm = (-ponderationb+ponderationf).clip(min=0)


        ponderation_segm = ponderation_segm/np.max(ponderation_segm)

        bezierContrast(ponderation_segm)
        ponderation_segm = (ponderation_segm).clip(min=0, max=1)
        ret, ponderation_segm=cv2.threshold(ponderation_segm,0.5,1,cv2.THRESH_BINARY)
        ponderationf = ponderationf.clip(min=0, max=1)
        ponderationb = ponderationb.clip(min=0, max=1)

        cv2.imwrite(self.output+"/foreground_ponderation.png", ponderationf * 255)
        cv2.imwrite(self.output+"/foreground_mask.png", maskf.astype(np.uint8))
        cv2.imwrite(self.output+"/background_ponderation.png", ponderationb * 255)
        cv2.imwrite(self.output+"/background_mask.png", maskb.astype(np.uint8))
        cv2.imwrite(self.output+"/segmentation_ponderation.png", ponderation_segm * 255)

        return ponderation_segm
    
    def infiere(self, mask):
        try:
            os.mkdir(self.output + "/foreground", 0x755 );
            os.mkdir(self.output + "/background", 0x755 );
           
        except:
            pass

        for file in self.inference_images:
            image = cv2.imread(file)
            image2 = image.copy()
            filterImage(image2 ,mask)
            
            name = file.split("\\")[-1]
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            count = countPixels(image2, self.mask_threshold)
            if count > self.pixel_count:
                cv2.imwrite(self.output+"/foreground/"+str(count)+name, image2)
                #cv2.imwrite(self.output+"/foreground/"+name, image)

            else:
                cv2.imwrite(self.output+"/background/"+str(count)+name, image2)
                #cv2.imwrite(self.output+"/background/"+name, image)


    def segmentImages(self, image_files, mask):
        for file in image_files:
            image = cv2.imread(file)
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    sphericals = self.sphericalCoordinates(image[y][x])
                    coord_x = int(np.round(sphericals[0]))
                    coord_y = int(np.round(sphericals[1]))
                    image[y][x] =  image[y][x] * mask[coord_y][coord_x] 
            
            name = file.split("\\")[-1]
            
            cv2.imwrite(self.output+"/"+name, image)


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Segmentation Sample")
        parser.add_argument('-ipf', '--images_path_foreground', type=str, default=r'D:\transfer_learning\losses', help="path to the images with foreground")
        parser.add_argument('-ipb', '--images_path_background', type=str, default=r'D:\transfer_learning\confusion', help="path to the images with background")
        parser.add_argument('-o', '--output', type=str, default=r'D:\output', help="results output")
        parser.add_argument('-ip', '--inference_path', type=str, default=r'D:\transfer_learning\losses', help="path to images to process. will split in two folders")
       #parser.add_argument('-mp', '--mask_path', type=str, default=r'D:\output\segmentation_ponderation.png', help="path to images to process. will split in two folders")
        parser.add_argument('-mp', '--mask_path', type=str, default=r'', help="path to images to process. will split in two folders")
        parser.add_argument('-th', '--mask_threshold', type=int, default=64, help="threshold for the segmentation")
        parser.add_argument('-pc', '--pixel_count', type=int, default=1000, help="pixel count to consider a foreground image")

        args = parser.parse_args()
        print(args)
        segmentation = Segmentation(args.images_path_foreground, args.images_path_background, args.inference_path, args.mask_path, args.output, args.mask_threshold, args.pixel_count)
        fullColorMask(args.output)
        mask=None
        if args.mask_path == r'':
            segmentation.loadSegmentationImages()
            mask = segmentation.computeMask()
            segmentImages(args.output, segmentation.foreground_images, mask)
            segmentImages(args.output, segmentation.background_images, mask)

        else:
            #loadMask 
            mask = cv2.imread(args.mask_path)            
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = mask / 255

        if mask is not None:
            segmentation.loadinferenceImages()
            segmentation.infiere(mask)