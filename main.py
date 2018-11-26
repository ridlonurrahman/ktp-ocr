import cv2

from vision_api_ocr import VisionAPIOCRLocationBased

# import image
driver_id = 541436371
image = cv2.imread('../stage-one/data/ktps/{}.jpg'.format(driver_id))

# create api object
api = VisionAPIOCRLocationBased(image)

# get text
text = api.get_text()
print(text)