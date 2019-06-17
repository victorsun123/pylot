import time
import cv2
import numpy as np
import math
from collections import namedtuple
from itertools import groupby

from erdos.message import Message
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

from pylot.utils import add_timestamp, create_detected_lane_stream
from pylot.utils import is_camera_stream

Line = namedtuple("Line", "x1, y1, x2, y2, slope")


class CannyEdgeLaneDetectionOperator(Op):
    def __init__(self,
                 name,
                 output_stream_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(CannyEdgeLaneDetectionOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._output_stream_name = output_stream_name
        self._kernel_size = 7

    @staticmethod
    def setup_streams(input_streams,
                      output_stream_name,
                      camera_stream_name=None):
        # Select camera input streams.
        camera_streams = input_streams.filter(is_camera_stream)
        if camera_stream_name:
            # Select only the camera the operator is interested in.
            camera_streams = camera_streams.filter_name(camera_stream_name)
        # Register a callback on the camera input stream.
        camera_streams.add_callback(
            CannyEdgeLaneDetectionOperator.on_msg_camera_stream)
        return [create_detected_lane_stream(output_stream_name)]

    def region_of_interest(self, image, points):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, points, 255)
        return cv2.bitwise_and(image, mask)

    def extrapolate_lines(self, image, left_line, right_line):
        top_y = None
        if left_line is not None and right_line is not None:
            top_y = min(
                [left_line.y1, left_line.y2, right_line.y1, right_line.y2])
        base_y = image.shape[0]

        final_lines = []
        if left_line is not None:
            actual_slope = float(left_line.y2 -
                                 left_line.y1) / float(left_line.x2 -
                                                       left_line.x1)
            base_x = int((base_y - left_line.y1) / actual_slope) + left_line.x1
            final_lines.append(
                Line(base_x, base_y, left_line.x1, left_line.y1, actual_slope))

            if top_y is None:
                top_y = min([left_line.y1, left_line.y2])

            top_x = int((top_y - left_line.y2) / actual_slope) + left_line.x2
            final_lines.append(
                Line(top_x, top_y, left_line.x2, left_line.y2, actual_slope))

        if right_line is not None:
            actual_slope = float(right_line.y2 -
                                 right_line.y1) / float(right_line.x2 -
                                                        right_line.x1)
            base_x = int(
                (base_y - right_line.y1) / actual_slope) + right_line.x1
            final_lines.append(
                Line(base_x, base_y, right_line.x1, right_line.y1,
                     actual_slope))

            if top_y is None:
                top_y = min([right_line.y1, right_line.y2])

            top_x = int((top_y - right_line.y2) / actual_slope) + right_line.x2
            final_lines.append(
                Line(top_x, top_y, right_line.x2, right_line.y2, actual_slope))
        return final_lines

    def draw_lines(self, image):
        lines = cv2.HoughLinesP(image,
                                rho=1,
                                theta=np.pi / 180.0,
                                threshold=40,
                                minLineLength=10,
                                maxLineGap=30)
        line_img = np.zeros((image.shape[0], image.shape[1], 3),
                            dtype=np.uint8)

        if lines is None:
            return line_img

        # Construct the Line tuple collection.
        cmp_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = math.degrees(math.atan2(y2 - y1, x2 - x1))
                cmp_lines.append(Line(x1, y1, x2, y2, slope))

        # Sort the lines by their slopes after filtering lines whose slopes
        # are > 20 or < -20.
        cmp_lines = sorted(
            filter(lambda line: line.slope > 20 or line.slope < -20,
                   cmp_lines),
            key=lambda line: line.slope)

        if len(cmp_lines) == 0:
            return line_img

        # Filter the lines with a positive and negative slope and choose
        # a single line out of those.
        left_lines = [
            line for line in cmp_lines if line.slope < 0 and line.x1 < 300
        ]
        right_lines = [
            line for line in cmp_lines
            if line.slope > 0 and line.x1 > image.shape[1] - 300
        ]

        final_lines = []
        # Find the longest line from the left and the right lines and
        # extrapolate to the middle of the image.
        left_line = None
        if len(left_lines) != 0:
            left_line = max(left_lines,
                            key=lambda line: abs(line.y2 - line.y1))
            final_lines.append(left_line)

        right_line = None
        if len(right_lines) != 0:
            right_line = max(right_lines,
                             key=lambda line: abs(line.y2 - line.y1))
            final_lines.append(right_line)

        final_lines.extend(self.extrapolate_lines(image, left_line,
                                                  right_line))

        for x1, y1, x2, y2, slope in final_lines:
            cv2.line(line_img, (x1, y1), (x2, y2),
                     color=(255, 0, 0),
                     thickness=2)
            cv2.putText(line_img, "({}, {})".format(x1, y1), (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2,
                        cv2.LINE_AA)
        return line_img

    def on_msg_camera_stream(self, msg):
        start_time = time.time()
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        # Make a copy of the image coming into the operator.
        image = np.copy(msg.frame)

        # Get the dimensions of the image.
        x_lim, y_lim = image.shape[1], image.shape[0]

        # Convert to grayscale.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur.
        image = cv2.GaussianBlur(image, (self._kernel_size, self._kernel_size),
                                 0)

        # Apply the Canny Edge Detector.
        image = cv2.Canny(image, 30, 60)

        # Define a region of interest.
        points = np.array(
            [[
                (0, y_lim),  # Bottom left corner.
                (0, y_lim - 60),
                (x_lim // 2 - 20, y_lim // 2),
                (x_lim // 2 + 20, y_lim // 2),
                (x_lim, y_lim - 60),
                (x_lim, y_lim),  # Bottom right corner.
            ]],
            dtype=np.int32)
        image = self.region_of_interest(image, points)

        # Hough lines.
        image = self.draw_lines(image)

        if self._flags.visualize_lane_detection:
            final_img = np.copy(msg.frame)
            add_timestamp(msg.timestamp, final_img)
            final_img = cv2.addWeighted(final_img, 0.8, image, 1.0, 0.0)
            cv2.imshow(self.name, final_img)
            cv2.waitKey(1)

        output_msg = Message(image, msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_msg)

    def execute(self):
        self.spin()
