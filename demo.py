from google.cloud import vision
from pyaxidraw import axidraw
from pyfirmata import Arduino, util
from rnn import rnn
import cv2
import drawing
import io
import logging
import numpy as np
import openai
import os
import os
import re
import svgwrite
import time

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

class Hand(object):

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.nn = rnn(
            log_dir='logs',
            checkpoint_dir='checkpoints',
            prediction_dir='predictions',
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None, initial_ypos=None):
        valid_char_set = set(drawing.alphabet)
        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    (
                        "Each line must be at most 75 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

        strokes = self._sample(lines, biases=biases, styles=styles)
        self._draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths, initial_ypos=initial_ypos)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 50*max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5]*num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load('styles/style-{}-strokes.npy'.format(style))
                c_p = np.load('styles/style-{}-chars.npy'.format(style)).tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

    def _draw(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None, initial_ypos=None):
        stroke_colors = stroke_colors or ['black']*len(lines)
        stroke_widths = stroke_widths or [2]*len(lines)

        line_height = 60
        view_width = 900
        view_height = 550

        dwg = svgwrite.Drawing(filename=filename, size=('8.5in', '5.5in'))
        dwg.viewbox(width=view_width, height=view_height)

        initial_ypos = initial_ypos or (3*line_height / 4)
        initial_coord = np.array([0, -initial_ypos])
        for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):

            if not line:
                initial_coord[1] -= line_height
                continue

            offsets[:, :2] *= 1.5
            strokes = drawing.offsets_to_coords(offsets)
            strokes = drawing.denoise(strokes)
            strokes[:, :2] = drawing.align(strokes[:, :2])

            strokes[:, 1] *= -1
            strokes[:, :2] -= strokes[:, :2].min() + initial_coord
            strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*strokes.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
                if eos==1.0:
                    path = svgwrite.path.Path(p)
                    path = path.stroke(color=color, width=width, linecap='round').fill("none")
                    dwg.add(path)
                    p = "M{},{} ".format(0, 0)

            initial_coord[1] -= line_height

        dwg.save()

def get_image_from_webcam():
    rectangle = (350,475), (1250,1025)
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(1)
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        rect = cv2.rectangle(frame, rectangle[0], rectangle[1], (255,0,0), 2)
        cv2.imshow("preview", rect)
        cv2.waitKey(1)
        rval, frame = vc.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        break
    
    cv2.imwrite("./webcam.jpg", frame[rectangle[0][1]:rectangle[1][1], rectangle[0][0]:rectangle[1][0]])
    vc.release()
    # cv2.destroyWindow("preview")
    return frame

def detect_text():
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()
    get_image_from_webcam()

    with io.open('./webcam.jpg', 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    texts = response.text_annotations
    if(len(texts)):
        cv2.namedWindow("text")
        frame = cv2.imread('webcam.jpg')
        vertices = texts[0].bounding_poly.vertices
        rectangle = (vertices[0].x, vertices[0].y),(vertices[2].x, vertices[2].y)
        rectangleImg = cv2.rectangle(frame, rectangle[0], rectangle[1], (0,255,0), 2)
        cv2.imshow("text", rectangleImg)
        cv2.waitKey(10)
        return texts[0].description, rectangle
    raise Exception("No text detected")

def get_pen_in(sensor):
    value = sensor.read()
    if value is None or value < 0.5:
        print("pen is in", value)
        return True
    print("pen is out", value)
    return False

if __name__ == '__main__':
    # initialize everything
    state = "ROBOT_WAITING"
    ad = axidraw.AxiDraw()
    ad.plot_setup()
    ad.options.mode = "align"
    ad.plot_run()
    board = Arduino('/dev/cu.usbserial-0001')
    it = util.Iterator(board)
    it.start()
    photoresistor = board.analog[5]
    photoresistor.enable_reporting()

    while True:
        pen_in = get_pen_in(photoresistor)
        if state == "ROBOT_WAITING":
            time.sleep(1)
            get_image_from_webcam()
            print("waiting...")
            if pen_in is False and get_pen_in(photoresistor):
                state = "ROBOT_THINKING"

        if state == "ROBOT_THINKING":
            human_input, bounding_box = detect_text()
            human_input = human_input.replace('\n', ' ').replace('\r', '').lower()
            print("Detected:", human_input)
            print("Querying OpenAI...")
            response = openai.Completion.create(model="text-davinci-002", prompt=human_input, temperature=0.25, max_tokens=50)
            robot_output = response.choices[0].text
            print("OpenAI response:", robot_output)
            robot_output = re.sub(r"[^%s]" % ''.join(drawing.alphabet), "", robot_output.replace('\n', '. '))
            hand = Hand()
            print("writing...")
            words = robot_output.split()
            lines = [' '.join(linewords) for linewords in np.array_split(words, len(words)//5)] if len(words) > 5 else [robot_output]
            biases = [.95 for line in lines]
            styles = [4 for line in lines]
            stroke_colors = ['black' for line in lines]
            stroke_widths = [1 for line in lines]
            print("Synthesizing handwriting...")
            hand.write(
                filename='img/usage_demo.svg',
                lines=lines,
                biases=biases,
                styles=styles,
                stroke_colors=stroke_colors,
                stroke_widths=stroke_widths,
                initial_ypos=bounding_box[1][1]+40
            )
            state = "ROBOT_WRITING"

        if state == "ROBOT_WRITING":
            print("Writing to plotter: ", )
            ad.options.mode = "plot"
            ad.plot_setup("img/usage_demo.svg")
            ad.options.pen_pos_down = 30
            ad.plot_run()

            ad.plot_setup()
            ad.options.mode = "align"
            ad.plot_run()
            state = "ROBOT_WAITING"