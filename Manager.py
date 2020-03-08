from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import os


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    browser = webdriver.Chrome(ChromeDriverManager().install())

    browser.get(os.path.join(path, "index.html"))

    # elem = browser.find_element_by_name('p')  # Find the search box
    # browser.quit()
