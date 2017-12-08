from time import sleep
from .cv_helper import surf_and_flann_get_object_coordinate, compare_diff
from datetime import datetime, timedelta
from os.path import join

config = {}


def setup(appium_driver, image_dir):
    """you should always setup before using methods in this module."""
    config['driver'] = appium_driver
    config['dir'] = image_dir
    config['screen'] = join(image_dir, 'current_screen.png')


def click_image(image_name, sleep_time=3):
    """method to click an image on device."""
    sleep(sleep_time)
    driver = config['driver']

    full_path = join(config['dir'], image_name)
    driver.save_screenshot(config['screen'])

    driver_h = driver.get_window_size()['height']
    driver_w = driver.get_window_size()['width']
    print('windows size height, width: {}, {}'.format(driver_h, driver_w))
    x, y, h, w = surf_and_flann_get_object_coordinate(full_path, config['screen'])

    scale_x = driver_w / w
    scale_y = driver_h / h

    final_x = x * scale_x
    final_y = y * scale_y
    driver.tap([(final_x, final_y)])
    sleep(2)


def wait_for_screenshot(screenshot_name, timeout=10):
    """
    for waiting for screenshot, you can design your own algorithm to determine exit criteria.
    ssim will give you info of the diff, so you may need to update compare_diff method to
    achieve your objective
    """

    driver = config['driver']
    start = datetime.now()
    end = start + timedelta(seconds=timeout)
    target_screen_path = join(config['dir'], screenshot_name)

    while datetime.now() < end:
        driver.save_screenshot(config['screen'])
        a, b, score = compare_diff(target_screen_path, config['screen'])
        if score > 0.95:
            return

    raise TimeoutError('timeout to wait for: {}'.format(screenshot_name))
