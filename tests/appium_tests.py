from appium import webdriver
from utils.cv_engine import setup, click_image, wait_for_screenshot
from os.path import join, dirname

desired_caps = {}
desired_caps['platformName'] = 'Android'
desired_caps['platformVersion'] = '7.1.1'
desired_caps['deviceName'] = 'device'
# desired_caps['appPackage'] = 'net.oneplus.launcher'
desired_caps['appPackage'] = 'com.tencent.tmgp.sgame'
# desired_caps['appActivity'] = 'DialtactsActivity'
desired_caps['noRest'] = True
desired_caps['autoLaunch'] = False

driver = webdriver.Remote('http://localhost:4723/wd/hub', desired_caps)
setup(appium_driver=driver, image_dir=join(dirname(dirname(__file__)), 'images'))
click_image('start-game.png')


# driver.tap([(900,1800)])
# driver.find_element_by_id('com.android.dialer:id/search_box_collapsed').click()
# search_box = driver.find_element_by_id('com.android.dialer:id/search_view')
# search_box.click()
# search_box.send_keys('hello toby')
