import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from collections import namedtuple
import cv2
import carla
import time
import random

Threshold = namedtuple("Threshold", "min, max")

thresholds = {
    "gradients": Threshold(min=40, max=130),
    "saturation": Threshold(min=30, max=100),
    "direction": Threshold(min=0.5, max=1),
}

counter = 0

def find_lane_pixels(binary_warped):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peaks of the left and right lanes.
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] // 2)
    peaks = scipy.signal.find_peaks(histogram, height=10)[0]
    if len(peaks) > 1:
        leftx_base, rightx_base = peaks[0], peaks[-1]
    else:
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 20
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 200

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    #lefty = []
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            left_lane_inds.append(good_left_inds)
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            #left_lane_inds.append(np.mean(nonzerox[good_left_inds]))
            #lefty.append(win_y_low + (window_height / 2))
        if len(good_right_inds) > minpix:
            right_lane_inds.append(good_right_inds)
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    #right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    #leftx = left_lane_inds
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ## Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Plot the points.
    for x, y in zip(left_fitx, ploty):
        cv2.circle(out_img, (int(x), int(y)), 1, (0, 0, 255))

    return out_img, ploty, left_fitx, right_fitx


def process_images(msg):
    global counter
    # Convert the BGRA image to BGR.
    image = np.frombuffer(msg.raw_data, dtype=np.dtype('uint8'))
    image = np.reshape(image, (msg.height, msg.width, 4))[:, :, :3]
    cv2.imshow("Image", image)

    # Convert the image from BGR to HLS.
    s_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 2]

    # Apply the Sobel operator in the x-direction.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)

    # Apply the Sobel operator in the y-direction.
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

    # Get the absolute values of x, y and xy gradients.
    abs_sobelx, abs_sobely = np.absolute(sobelx), np.absolute(sobely)
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)

    # Threshold the magnitude of the gradient.
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > thresholds['gradients'].min)
             & (scaled_sobel < thresholds['gradients'].max)] = 1

    # Threshold the color channel.
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > thresholds['saturation'].min)
             & (s_channel < thresholds['saturation'].max)] = 1

    color_binary = np.dstack(
        (np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Take a bitwise or of all our heuristics.
    final_image = np.zeros_like(sxbinary)
    final_image[(s_binary == 1) | (sxbinary == 1)] = 255

    # Convert the image to a bird's eye view.
    #source_points = np.float32([[30, msg.height], [353, 332], [440, 332],
    #                            [780, msg.height]])
    source_points = np.float32([[163, 500], [353, 332], [440, 332], [636, 500]])
    offset = 100
    #destination_points = np.float32([[offset, msg.height],
    #                                 [offset, msg.height // 2],
    #                                 [msg.width - offset, msg.height // 2],
    #                                 [msg.width - offset, msg.height]])
    destination_points = np.float32([[offset, msg.height],
                                     [offset, 0],
                                     [msg.width - offset, 0],
                                     [msg.width - offset, msg.height]])
    M = cv2.getPerspectiveTransform(source_points, destination_points)
    M_inv = cv2.getPerspectiveTransform(destination_points, source_points)
    warped_image = cv2.warpPerspective(final_image, M, (msg.width, msg.height))

    # Fit the polynomial.
    fit_img, ploty, left_fit, right_fit = fit_polynomial(warped_image)
    cv2.imshow("Fitted image", fit_img)
    #filled_image = np.zeros_like(warped_image).astype(np.uint8)
    #color_filled = np.dstack((filled_image, filled_image, filled_image))
    #left_points = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    #right_points = np.array(
    #    [np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    #points = np.hstack((left_points, right_points))
    #cv2.fillPoly(color_filled, np.int_([points]), (0, 255, 0))
    #unwarped = cv2.warpPerspective(color_filled, M_inv,
    #                               (msg.width, msg.height))
    #result = cv2.addWeighted(image, 1, unwarped, 0.3, 0)

    #cv2.imshow(
    #    "Fitted Image",
    #    cv2.addWeighted(np.dstack((warped_image, warped_image, warped_image)),
    #                    1, color_filled, 0.3, 0))
    #cv2.imwrite('warped/image{}.png'.format(counter), warped_image)
    #cv2.imshow("Warped Image", warped_image)
    counter += 1


def spawn_driving_vehicle(client, world):
    """ This function spawns the driving vehicle and puts it into
    an autopilot mode.

    Args:
        client: The carla.Client instance representing the simulation to
          connect to.
        world: The world inside the current simulation.

    Returns:
        A carla.Actor instance representing the vehicle that was just spawned.
    """
    # Get the blueprint of the vehicle and set it to AutoPilot.
    vehicle_bp = random.choice(
        world.get_blueprint_library().filter('vehicle.*'))
    while not vehicle_bp.has_attribute('number_of_wheels') or not int(
            vehicle_bp.get_attribute('number_of_wheels')) == 4:
        vehicle_bp = random.choice(
            world.get_blueprint_library().filter('vehicle.*'))
    vehicle_bp.set_attribute('role_name', 'autopilot')

    # Get the spawn point of the vehicle.
    start_pose = random.choice(world.get_map().get_spawn_points())

    # Spawn the vehicle.
    batch = [
        carla.command.SpawnActor(vehicle_bp, start_pose).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True))
    ]
    vehicle_id = client.apply_batch_sync(batch)[0].actor_id

    # Find the vehicle and return the carla.Actor instance.
    time.sleep(
        0.5)  # This is so that the vehicle gets registered in the actors.
    return world.get_actors().find(vehicle_id)


def spawn_rgb_camera(world, location, rotation, vehicle):
    """ This method spawns an RGB camera with the default parameters and the
    given location and rotation. It also attaches the camera to the given
    actor.

    Args:
        world: The world inside the current simulation.
        location: The carla.Location instance representing the location where
          the camera needs to be spawned with respect to the vehicle.
        rotation: The carla.Rotation instance representing the rotation of the
          spawned camera.
        vehicle: The carla.Actor instance to attach the camera to.

    Returns:
        An instance of the camera spawned in the world.
    """
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    transform = carla.Transform(location=location, rotation=rotation)
    return world.spawn_actor(camera_bp, transform, attach_to=vehicle)


def main():
    # Connect to the CARLA instance.
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # Spawn the vehicle.
    vehicle = spawn_driving_vehicle(client, world)

    # Spawn the camera and register a function to listen to the images.
    camera = spawn_rgb_camera(world, carla.Location(x=2.0, y=0.0, z=1.4),
                              carla.Rotation(roll=0, pitch=0, yaw=0), vehicle)
    camera.listen(process_images)

    return vehicle, camera, world


if __name__ == "__main__":
    vehicle, camera, world = main()
    try:
        while True:
            time.sleep(1 / 100.0)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        # Destroy the actors.
        vehicle.destroy()
        camera.destroy()
