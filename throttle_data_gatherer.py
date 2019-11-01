from absl import app
from absl import flags

import erdos.graph
from erdos.op import Op
from erdos.timestamp import Timestamp
from erdos.utils import setup_logging
import pylot.config
from pylot.control.messages import ControlMessage
import pylot.operator_creator
import pylot.simulation.utils
import pylot.utils
import numpy as np

FLAGS = flags.FLAGS


class SynchronizerOp(Op):
    def __init__(self, name):
        super(SynchronizerOp, self).__init__(name)

    @staticmethod
    def setup_streams(input_streams):
        input_streams.add_completion_callback(SynchronizerOp.on_watermark)
        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        return [pylot.utils.create_control_stream()]

    def on_watermark(self, msg):
        control_msg = ControlMessage(
            0, 0, 0, False, False, msg.timestamp)
        self.get_output_stream('control_stream').send(control_msg)


class ThrottleOp(Op):
    def __init__(self, name):
        super(ThrottleOp, self).__init__(name)
        self._logger = setup_logging(self.name, "throttle_log.log")
        self._ground_vehicle_id = None
        _, self._world = pylot.simulation.carla_utils.get_world(FLAGS.carla_host,
                                                                FLAGS.carla_port,
                                                                FLAGS.carla_timeout)
        self.log_count = 0
        self.data = []

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_ground_vehicle_id_stream).add_callback(
            ThrottleOp.on_ground_vehicle_id_update)
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            ThrottleOp.on_can_bus_update)
        return [pylot.utils.create_control_stream()]

    def on_ground_vehicle_id_update(self, msg):
        self._logger.info("Set Ground Vehicle Id: {}".format(msg.data))
        self._ground_vehicle_id = msg.data

    def on_can_bus_update(self, msg):
        self._logger.info("Can Bus Msg: {}".format(msg))
        # sample every 1 second (10fps / 10frames = 1s)
        if self._ground_vehicle_id and self.log_count % 10 == 0:
            ego_vehicle = self._world.get_actor(self._ground_vehicle_id)
            control = ego_vehicle.get_control()
            velocity = ego_vehicle.get_velocity()
            acceleration = ego_vehicle.get_acceleration()

            throttle = control.throttle
            steer = control.steer
            brake = control.brake
            speed = np.sqrt(velocity.x**2 + velocity.y**2)
            accel = np.sqrt(acceleration.x**2 + acceleration.y**2)

            self.data.append([[accel, speed, steer, brake, throttle]])

        # save every 1000 points
        if self.log_count % 1000 == 0 and self.log_count > 0 and len(self.data) > 0:
            self._logger.info("Logging data of length: {}".format(len(self.data)))
            np.save("./data/{}.npy".format(msg.timestamp), self.data)
            self.data = []

        self.log_count += 1
        control_msg = ControlMessage(
            0, 0, 0, False, False, msg.timestamp)
        self.get_output_stream('control_stream').send(control_msg)


def main(argv):
    # Define graph
    graph = erdos.graph.get_current_graph()

    # Add operator that interacts with the Carla simulator.
    (carla_op, _, _) = pylot.operator_creator.create_driver_ops(
        graph, [], [], auto_pilot=FLAGS.carla_auto_pilot)

    if FLAGS.carla_auto_pilot:
        throttle_op = graph.add(ThrottleOp, name='throttle_op')
        sync_op = graph.add(SynchronizerOp, name='sync_op')
        graph.connect([carla_op], [throttle_op])
        graph.connect(
            [throttle_op],
            [sync_op])
        graph.connect([sync_op], [carla_op])
        graph.connect([throttle_op], [carla_op])

    graph.execute(FLAGS.framework)


if __name__ == '__main__':
    app.run(main)
