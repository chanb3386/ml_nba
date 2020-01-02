from selenium import webdriver
import time


# function that updates team stats and opp stats files for most recent season
# WIP
def updateData():
    url = "https://www.basketball-reference.com/leagues/NBA_2020.html"
    driver = webdriver.Chrome('../cdriver/chromedriver')
    driver.get(url)
    time.sleep(5)
    driver.quit()
    return


updateData()
