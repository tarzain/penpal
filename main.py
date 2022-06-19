import cv2
from google.cloud import vision
import io

def get_image_from_webcam():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break
    cv2.imwrite("./webcam.jpg", frame)
    vc.release()
    cv2.destroyWindow("preview")
    return frame

def detect_text():
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()
    get_image_from_webcam()

    with io.open('./webcam.jpg', 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')
    print(texts[0].description)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == "__main__":
    detect_text()