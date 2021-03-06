from collections import deque

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot.perception.detection.utils import get_precision_recall_at_iou
from pylot.utils import is_obstacles_stream


class DetectionEvalGroundOperator(Op):
    def __init__(self, name, flags, log_file_name=None, csv_file_name=None):
        super(DetectionEvalGroundOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
        self._ground_bboxes = deque()
        self._iou_thresholds = [0.1 * i for i in range(1, 10)]

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(is_obstacles_stream).add_callback(
            DetectionEvalGroundOperator.on_ground_obstacles)
        return []

    def on_ground_obstacles(self, msg):
        # Ignore the first several seconds of the simulation because the car is
        # not moving at the beginning.
        game_time = msg.timestamp.coordinates[0]
        bboxes = []
        # Select the pedestrian bounding boxes.
        for det_obj in msg.detected_objects:
            if det_obj.label == 'pedestrian':
                bboxes.append(det_obj.corners)

        # Remove the buffered bboxes that are too old.
        while (len(self._ground_bboxes) > 0 and
               game_time - self._ground_bboxes[0][0] >
               self._flags.eval_ground_truth_max_latency):
            self._ground_bboxes.popleft()

        for (old_game_time, old_bboxes) in self._ground_bboxes:
            # Ideally, we would like to take multiple precision values at
            # different recalls and average them, but we can't vary model
            # confidence, so we just return the actual precision.
            if (len(bboxes) > 0 or len(old_bboxes) > 0):
                latency = game_time - old_game_time
                precisions = []
                for iou in self._iou_thresholds:
                    (precision, _) = get_precision_recall_at_iou(
                        bboxes, old_bboxes, iou)
                    precisions.append(precision)
                self._logger.info("Precision {}".format(precisions))
                avg_precision = float(sum(precisions)) / len(precisions)
                self._logger.info(
                    "The latency is {} and the average precision is {}".format(
                        latency, avg_precision))
                self._csv_logger.info('{},{},{},{}'.format(
                    time_epoch_ms(), self.name, latency, avg_precision))

        # Buffer the new bounding boxes.
        self._ground_bboxes.append((game_time, bboxes))
