import os
from collections import namedtuple

import carla
import logging
from absl import app
from absl import flags

import erdos.graph
import pylot.config
import pylot.simulation.utils

from pylot.simulation.camera_driver_operator import CameraDriverOperator
from pylot.simulation.carla_scenario_operator import CarlaScenarioOperator
from pylot.simulation.perfect_pedestrian_detector_operator import \
        PerfectPedestrianDetectorOperator
from pylot.simulation.carla_utils import get_world
from pylot.simulation.perfect_planning_operator import PerfectPlanningOperator
from pylot.perception.detection.detection_operator import DetectionOperator

FLAGS = flags.FLAGS

Camera = namedtuple("Camera", "camera_setup, instance")


def add_camera_operator(graph):
    """ Adds the RGB and depth camera operator to the given graph and returns
    the setups and the instances of the operators added to the graph.

    Args:
        graph: The erdos.graph instance to add the operator to.

    Returns:
        A tuple containing the RGB and depth camera instances, which consists
        of the setup used to spawn the camera, and the operator instance
        itself.
    """
    # Create the camera setup needed to add the operator to the grpah.
    camera_location = pylot.simulation.utils.Location(1.5, 0.0, 1.4)
    camera_rotation = pylot.simulation.utils.Rotation(0, 0, 0)
    camera_transform = pylot.simulation.utils.Transform(
        camera_location, camera_rotation)

    # Create the RGB camera and add the operator to the graph.
    rgb_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.CENTER_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        camera_transform,
        fov=90)

    rgb_camera_operator = graph.add(
        CameraDriverOperator,
        name=rgb_camera_setup.name,
        init_args={
            'camera_setup': rgb_camera_setup,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        },
        setup_args={'camera_setup': rgb_camera_setup})
    rgb_camera = Camera(camera_setup=rgb_camera_setup,
                        instance=rgb_camera_operator)

    # Create the depth camera and add the operator to the graph.
    depth_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.DEPTH_CAMERA_NAME,
        'sensor.camera.depth',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        camera_transform,
        fov=90)
    depth_camera_operator = graph.add(
        CameraDriverOperator,
        name=depth_camera_setup.name,
        init_args={
            'camera_setup': depth_camera_setup,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        },
        setup_args={'camera_setup': depth_camera_setup})
    depth_camera = Camera(camera_setup=depth_camera_setup,
                          instance=depth_camera_operator)

    return (rgb_camera, depth_camera)


def add_carla_operator(graph):
    """ Adds the Carla operator to the given graph and returns the graph.

    Args:
        graph: The erdos.graph instance to add the operator to.

    Returns:
        The operator instance depicting the Carla operator returned by the
        graph add method.
    """
    carla_operator = graph.add(CarlaScenarioOperator,
                               name='carla',
                               init_args={
                                   'role_name': 'hero',
                                   'flags': FLAGS,
                                   'log_file_name': FLAGS.log_file_name,
                               })
    return carla_operator


def set_asynchronous_mode(world):
    """ Sets the simulator to asynchronous mode.

    Args:
        world: The world instance of the simulator to set the asynchronous
            mode on.
    """
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)


def add_perfect_pedestrian_detector_operator(graph, camera_setup,
                                             output_stream_name):
    """ Adds the perfect pedestrian detector operator to the graph, and returns
    the added operator.

    Args:
        graph: The erdos.graph instance to add the operator to.
        camera_setup: The camera setup to use for projecting the pedestrians
            onto the view of the camera.
        output_stream_name: The name of the output stream where the results are
            published.
    Returns:
        The operator instance depicting the PerfectPedestrianOperator returned
        by the graph add method.
    """
    pedestrian_detector_operator = graph.add(
        PerfectPedestrianDetectorOperator,
        name='perfect_pedestrian',
        init_args={
            'output_stream_name': output_stream_name,
            'camera_setup': camera_setup,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        },
        setup_args={'output_stream_name': output_stream_name})
    return pedestrian_detector_operator


def add_planning_operator(graph, destination, behavior, model_path, speed):
    """ Adds the perfect planning operator to the graph, and returns the added
    operator.

    Args:
        graph: The erdos.graph instance to add the operator to.
        destination: The destination to plan until.
        behavior: The behavior that the planning operator needs to exhibit in
            case of emergencies.
        model_path: The path of the model to be run at the perception operator.
    Returns:
        The operator instance depicting the PerfectPlanningOperator returned
        by the graph add method.
    """
    planning_operator = graph.add(
        PerfectPlanningOperator,
        name='perfect_planning',
        init_args={
            'goal':
            destination,
            'behavior':
            behavior,
            'flags':
            FLAGS,
            'log_file_name':
            FLAGS.log_file_name,
            'csv_file_name':
            'results/{model_name}/{model_name}_{speed}_distance.csv'.format(
                model_name=model_path.split('/')[-2], speed=speed),
        })
    return planning_operator


def add_model_perception_operator(graph, output_stream_name, model_path,
                                  speed):
    """ Adds the Perception operator to the graph that runs a tensorflow model.

    Args:
        graph: The erdos.graph instance to add the operator to.
        output_stream_name: The name of the output stream where the results
        are published.
        model_path: The path of the model to be run at the perception operator.
    Returns:
        The operator instance depicting the PerceptionOperator returned by the
        graph.add method.
    """
    perception_operator = graph.add(
        DetectionOperator,
        name='detection_operator',
        init_args={
            'output_stream_name':
            output_stream_name,
            'model_path':
            model_path,
            'flags':
            FLAGS,
            'log_file_name':
            FLAGS.log_file_name,
            'csv_file_name':
            'results/{model_name}/{model_name}_{speed}_runtimes.csv'.format(
                model_name=model_path.split('/')[-2], speed=speed),
        },
        setup_args={'output_stream_name': output_stream_name})
    return perception_operator


def main(args):
    # Connect an instance to the simulator to make sure that we can turn the
    # synchronous mode off after the script finishes running.
    client, world = get_world(FLAGS.carla_host, FLAGS.carla_port,
                              FLAGS.carla_timeout)
    if client is None or world is None:
        raise ValueError("There was an issue connecting to the simulator.")

    if not os.path.exists('./results'):
        os.mkdir('results')

    if not os.path.exists('./results/{}'.format(
            FLAGS.model_path.split('/')[-2])):
        os.mkdir('results/{}'.format(FLAGS.model_path.split('/')[-2]))

    try:
        # Define the ERDOS graph.
        graph = erdos.graph.get_current_graph()

        # Define the CARLA operator.
        carla_operator = add_carla_operator(graph)

        # Add the camera operator to the data-flow graph.
        rgb_camera, depth_camera = add_camera_operator(graph)
        graph.connect([carla_operator],
                      [rgb_camera.instance, depth_camera.instance])

        perception_output_stream = 'perfect_pedestrian_bboxes'
        object_detector_operator = None
        if FLAGS.use_perfect_perception:
            # Add a perfect pedestrian detector operator.
            object_detector_operator = add_perfect_pedestrian_detector_operator(
                graph, rgb_camera.camera_setup, perception_output_stream)
            graph.connect([carla_operator, depth_camera.instance],
                          [object_detector_operator])
        else:
            # Add the model-based perception operator.
            object_detector_operator = add_model_perception_operator(
                graph, perception_output_stream, FLAGS.model_path,
                FLAGS.target_speed)
            graph.connect([rgb_camera.instance], [object_detector_operator])

        # Add a perfect planning operator.
        perfect_planning_operator = add_planning_operator(
            graph, carla.Location(x=17.73, y=327.07, z=0.5), FLAGS.plan,
            FLAGS.model_path, FLAGS.target_speed)
        graph.connect([carla_operator, object_detector_operator],
                      [perfect_planning_operator])
        graph.connect([perfect_planning_operator], [carla_operator])

        # Change the logging level to warning.
        # logging.disable(logging.INFO)

        graph.execute(FLAGS.framework)
    except KeyboardInterrupt:
        set_asynchronous_mode(world)
    except Exception as e:
        set_asynchronous_mode(world)
        raise e


if __name__ == "__main__":
    app.run(main)
