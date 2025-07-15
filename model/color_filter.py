import cv2
import matplotlib.pyplot as plt

class color_filter:
    """This creates a class that takes some image
    and allows us to filter a binary image with sliding boundaries
    """
    def __init__(self):
        super().__init__()
        self.cv_image = None                        # the latest image from the camera
        self.red_lower_bound = 0
        self.green_lower_bound = 0
        self.blue_lower_bound = 0
        self.red_upper_bound = 255
        self.green_upper_bound = 255
        self.blue_upper_bound = 255

        cv2.namedWindow('video_window')
        cv2.namedWindow('binary_window')
        cv2.createTrackbar('red lower bound', 'binary_window', self.red_lower_bound, 255, self.set_red_lower_bound)
        cv2.createTrackbar('red upper bound', 'binary_window', self.red_upper_bound, 255, self.set_red_upper_bound)
        cv2.createTrackbar('green lower bound', 'binary_window', self.green_lower_bound, 255, self.set_green_lower_bound)
        cv2.createTrackbar('green upper bound', 'binary_window', self.green_upper_bound, 255, self.set_green_upper_bound)
        cv2.createTrackbar('blue lower bound', 'binary_window', self.blue_lower_bound, 255, self.set_blue_lower_bound)
        cv2.createTrackbar('blue upper bound', 'binary_window', self.blue_upper_bound, 255, self.set_blue_upper_bound)


    def set_red_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.red_lower_bound = val

    def set_red_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red upper bound """
        self.red_upper_bound = val

    def set_green_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the green lower bound """
        self.green_lower_bound = val

    def set_green_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the green upper bound """
        self.green_upper_bound = val

    def set_blue_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the blue lower bound """
        self.blue_lower_bound = val

    def set_blue_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the blue upper bound """
        self.blue_upper_bound = val
    
    def run(self):
        self.cv_image = cv2.imread('/home/alexiswu/Downloads/come_frame383.jpg')
        if not self.cv_image is None:
            self.binary_image = cv2.inRange(self.cv_image, (self.blue_lower_bound,self.green_lower_bound,self.red_lower_bound), (self.blue_upper_bound,self.green_upper_bound,self.red_upper_bound))
            cv2.imshow('video_window', self.cv_image)
            cv2.imshow('binary_window', self.binary_image)
        cv2.waitKey(5)

def main():
    node = color_filter()
    while 1:
        node.run()

if __name__ =='__main__':
    main()