from appium import webdriver

desired_caps = {}
desired_caps['platformName'] = 'Android'
desired_caps['platformVersion'] = '7.1.1'
desired_caps['deviceName'] = 'device'
desired_caps['appPackage'] = 'com.android.dialer'
desired_caps['appActivity'] = 'DialtactsActivity'

driver = webdriver.Remote('http://localhost:4723/wd/hub', desired_caps)
driver.tap([900,1800])
# driver.find_element_by_id('com.android.dialer:id/search_box_collapsed').click()
# search_box = driver.find_element_by_id('com.android.dialer:id/search_view')
# search_box.click()
# search_box.send_keys('hello toby')