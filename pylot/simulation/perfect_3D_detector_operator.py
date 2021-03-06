from collections import deque
import threading

from erdos.message import WatermarkMessage
from erdos.op import Op
from pylot.simulation.perfect_detector_operator import PerfectDetectorOp
from erdos.utils import setup_csv_logging, setup_logging

import pylot.utils
from pylot.perception.detection.utils import DetectedObject,\
    annotate_image_with_bboxes, save_image, visualize_image
from pylot.perception.messages import DetectorMessage
from pylot.simulation.utils import get_2d_bbox_from_3d_box
from pylot.simulation.utils import add_noise_3D_bounding_box
from pylot.simulation.carla_utils import get_world



class PerfectDetector3dOp(PerfectDetectorOp):
    """ Operator that transforms information it receives from Carla into
    perfect bounding boxes in 3D, built off 2D perfect detector

    """
    def __init__(self, name, output_stream_name, bgr_camera_setup, \
            flags,log_file_name=None, csv_file_name=None):
        print("using 3d")
        super(PerfectDetector3dOp,self).__init__(name,output_stream_name, bgr_camera_setup,flags,log_file_name, csv_file_name)
        print("finished init")

        @staticmethod
    def setup_streams(input_streams, output_stream_name):
        # Register a callback on depth frames data stream.
        input_streams.filter(pylot.utils.is_depth_camera_stream).add_callback(
            PerfectDetectorOp.on_depth_camera_update)
        # Register a callback on BGR frames data stream.
        input_streams.filter(pylot.utils.is_center_camera_stream).add_callback(
            PerfectDetectorOp.on_bgr_camera_update)
        # Register a callback on segmented frames data stream.
        input_streams.filter(
            pylot.utils.is_segmented_camera_stream).add_callback(
                PerfectDetectorOp.on_segmented_frame)
        # Register a callback on can bus messages data stream.
        input_streams.filter(
            pylot.utils.is_can_bus_stream).add_callback(
                PerfectDetectorOp.on_can_bus_update)
        # Register a callback to receive pedestrian updates from Carla.
        input_streams.filter(
            pylot.utils.is_ground_pedestrians_stream).add_callback(
                PerfectDetectorOp.on_pedestrians_update)
        # Register a callback to receive vehicle updates from Carla.
        input_streams.filter(
            pylot.utils.is_ground_vehicles_stream).add_callback(
                PerfectDetectorOp.on_vehicles_update)
        # Register a completion watermark callback. The callback is invoked
        # after all the messages with a given timestamp have been received.
        input_streams.add_completion_callback(
            PerfectDetector3dOp.on_notification)
        # Register a callback on traffic lights data stream.
        input_streams.filter(
            pylot.utils.is_ground_traffic_lights_stream).add_callback(
                PerfectDetectorOp.on_traffic_light_update)
        # Register a callback on speed limits data stream.
        input_streams.filter(
            pylot.utils.is_ground_speed_limit_signs_stream).add_callback(
                PerfectDetectorOp.on_speed_limit_signs_update)
        # Register a callback on stop signs data stream.
        input_streams.filter(
            pylot.utils.is_ground_stop_signs_stream).add_callback(
                PerfectDetectorOp.on_stop_signs_update)
        # Stream on which to output bounding boxes.
        return [pylot.utils.create_obstacles_stream(output_stream_name)]

    def __get_pedestrians(self, pedestrians, vehicle_transform, depth_array):
        """ Transforms pedestrians into detected objects.
        Args:
            pedestrians: List of Pedestrian objects.
            vehicle_transform: Ego-vehicle transform.
            depth_array: Depth frame taken at the time when pedestrians were
                         collected.
        """
        det_objs = []
        for pedestrian in pedestrians:
            bounding_box = pedestrian.bounding_box
            if self._flags.obj_detection_noise:
                bounding_box = add_noise_3D_bounding_box(bounding_box)
            bbox = get_2d_bbox_from_3d_box(
                depth_array, vehicle_transform, pedestrian.transform,
                bounding_box, self._bgr_transform,
                self._bgr_intrinsic, self._bgr_img_size, 1.5, 3.0)
            if bbox is not None:
                det_objs.append(DetectedObject(bbox, 1.0, 'pedestrian'))
        return det_objs

    def __get_vehicles(self, vehicles, vehicle_transform, depth_array):
        """ Transforms vehicles into detected objects.
        Args:
            vehicles: List of Vehicle objects.
            vehicle_transform: Ego-vehicle transform.
            depth_array: Depth frame taken at the time when pedestrians were
                         collected.
        """
        det_objs = []
        for vehicle in vehicles:
            bounding_box = vehicle.bounding_box
            if self._flags.obj_detection_noise:
                print("before", bounding_box.extent.x)
                bounding_box = add_noise_3D_bounding_box(bounding_box)
                print("after", bounding_box.extent.x)

            bbox = get_2d_bbox_from_3d_box(
                depth_array, vehicle_transform, vehicle.transform,
                bounding_box, self._bgr_transform, self._bgr_intrinsic,
                self._bgr_img_size, 3.0, 3.0)
            if bbox is not None:
                det_objs.append(DetectedObject(bbox, 1.0, 'vehicle'))
        return det_objs

    def on_notification(self, msg):
        # Pop the oldest message from each buffer.
        print("3D update")
        with self._lock:
            if not self.synchronize_msg_buffers(
                    msg.timestamp,
                    [self._depth_imgs, self._bgr_imgs, self._segmented_imgs,
                     self._can_bus_msgs, self._pedestrians, self._vehicles,
                     self._traffic_lights, self._speed_limit_signs,
                     self._stop_signs]):
                return
            depth_msg = self._depth_imgs.popleft()
            bgr_msg = self._bgr_imgs.popleft()
            segmented_msg = self._segmented_imgs.popleft()
            can_bus_msg = self._can_bus_msgs.popleft()
            pedestrians_msg = self._pedestrians.popleft()
            vehicles_msg = self._vehicles.popleft()
            traffic_light_msg = self._traffic_lights.popleft()
            speed_limit_signs_msg = self._speed_limit_signs.popleft()
            stop_signs_msg = self._stop_signs.popleft()

        self._logger.info('Timestamps {} {} {} {} {} {}'.format(
            depth_msg.timestamp, bgr_msg.timestamp, segmented_msg.timestamp,
            can_bus_msg.timestamp, pedestrians_msg.timestamp,
            vehicles_msg.timestamp, traffic_light_msg.timestamp))

        # The popper messages should have the same timestamp.
        assert (depth_msg.timestamp == bgr_msg.timestamp ==
                segmented_msg.timestamp == can_bus_msg.timestamp ==
                pedestrians_msg.timestamp == vehicles_msg.timestamp ==
                traffic_light_msg.timestamp)

        self._frame_cnt += 1
        if (hasattr(self._flags, 'log_every_nth_frame') and
            self._frame_cnt % self._flags.log_every_nth_frame != 0):
            # There's no point to run the perfect detector if collecting
            # data, and only logging every nth frame.
            output_msg = DetectorMessage([], 0, msg.timestamp)
            self.get_output_stream(self._output_stream_name).send(output_msg)
            self.get_output_stream(self._output_stream_name)\
                .send(WatermarkMessage(msg.timestamp))
            return
        depth_array = depth_msg.frame
        vehicle_transform = can_bus_msg.data.transform
            
        det_ped = self.__get_pedestrians(
            pedestrians_msg.pedestrians, vehicle_transform, depth_array)

        det_vec = self.__get_vehicles(
            vehicles_msg.vehicles, vehicle_transform, depth_array)

        det_traffic_lights = pylot.simulation.utils.get_traffic_light_det_objs(
            traffic_light_msg.traffic_lights,
            vehicle_transform * depth_msg.transform,
            depth_msg.frame,
            depth_msg.width,
            depth_msg.height,
            self._town_name,
            depth_msg.fov)

        det_speed_limits = pylot.simulation.utils.get_speed_limit_det_objs(
            speed_limit_signs_msg.speed_signs,
            vehicle_transform,
            vehicle_transform * depth_msg.transform,
            depth_msg.frame, depth_msg.width, depth_msg.height,
            depth_msg.fov, segmented_msg.frame)

        det_stop_signs = pylot.simulation.utils.get_traffic_stop_det_objs(
            stop_signs_msg.stop_signs,
            vehicle_transform * depth_msg.transform,
            depth_msg.frame, depth_msg.width, depth_msg.height, depth_msg.fov)

        det_objs = (det_ped + det_vec + det_traffic_lights +
                    det_speed_limits + det_stop_signs)

        # Send the detected obstacles.
        output_msg = DetectorMessage(det_objs, 0, msg.timestamp)

        self.get_output_stream(self._output_stream_name).send(output_msg)
        # Send watermark on the output stream because operators do not
        # automatically forward watermarks when they've registed an
        # on completion callback.
        self.get_output_stream(self._output_stream_name)\
            .send(WatermarkMessage(msg.timestamp))

        if (self._flags.visualize_ground_obstacles or
            self._flags.log_detector_output):

            annotate_image_with_bboxes(
                bgr_msg.timestamp, bgr_msg.frame, det_objs)
            if self._flags.visualize_ground_obstacles:
                visualize_image(self.name, bgr_msg.frame)
            if self._flags.log_detector_output:
                save_image(pylot.utils.bgr_to_rgb(bgr_msg.frame),
                           bgr_msg.timestamp,
                           self._flags.data_path,
                           'perfect-detector-3D')
