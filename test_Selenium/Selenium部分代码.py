import pytest
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
import os

class TestUntitled():
  def setup_method(self, method):
    self.driver = webdriver.Chrome()
    self.vars = {}
    self.timings = []

  def teardown_method(self, method):
    # 保存性能数据为 JSON 文件
    with open("selenium_performance_report.json", "w") as f:
      json.dump(self.timings, f, indent=2)
    self.driver.quit()

  def log_timing(self, name, action):
    start = time.time()
    action()
    duration = time.time() - start
    print(f"{name} took {duration:.2f} seconds")
    self.timings.append({"step": name, "duration": round(duration, 3)})

  def test_untitled(self):
    # 1. 记录页面加载时间
    start_time = time.time()
    self.driver.get("http://localhost:8080/")
    self.driver.set_window_size(802, 863)
    WebDriverWait(self.driver, 10).until(
        expected_conditions.presence_of_element_located((By.CSS_SELECTOR, ".home-mobile-hero-banner"))
    )
    page_load_time = time.time() - start_time
    print(f"Page Load Time: {page_load_time:.2f} seconds")
    self.timings.append({"step": "Load Home Page", "duration": round(page_load_time, 3)})

    # 2. 主页交互
    self.log_timing("Click Banner", lambda: self.driver.find_element(By.CSS_SELECTOR, ".home-mobile-hero-banner").click())

    # 3. 切换货币
    def select_currency():
      dropdown = self.driver.find_element(By.NAME, "currency_code")
      dropdown.click()
      dropdown.find_element(By.XPATH, "//option[. = 'JPY']").click()
    self.log_timing("Select Currency - JPY", select_currency)

    # 4. 添加第一个商品（3个）
    def select_first_product():
      self.driver.find_element(By.CSS_SELECTOR, ".col-md-4:nth-child(3) .hot-product-card-img-overlay").click()
      quantity_dropdown = self.driver.find_element(By.ID, "quantity")
      quantity_dropdown.click()
      quantity_dropdown.find_element(By.XPATH, "//option[. = '3']").click()
      self.driver.find_element(By.CSS_SELECTOR, ".cymbal-button-primary").click()
    self.log_timing("Add First Product (3 items)", select_first_product)

    self.log_timing("Continue Shopping 1", lambda: self.driver.find_element(By.LINK_TEXT, "Continue Shopping").click())

    # 5. 添加第二个商品（10个）
    def select_second_product():
      self.driver.find_element(By.CSS_SELECTOR, ".col-md-4:nth-child(5) .hot-product-card-img-overlay").click()
      quantity_dropdown = self.driver.find_element(By.ID, "quantity")
      quantity_dropdown.click()
      quantity_dropdown.find_element(By.XPATH, "//option[. = '10']").click()
      self.driver.find_element(By.CSS_SELECTOR, ".cymbal-button-primary").click()
    self.log_timing("Add Second Product (10 items)", select_second_product)

    self.log_timing("Continue Shopping 2", lambda: self.driver.find_element(By.LINK_TEXT, "Continue Shopping").click())

    # 6. 跳转某页面（点击"13"）
    self.log_timing("Click Page 13", lambda: self.driver.find_element(By.LINK_TEXT, "13").click())

    # 7. 点击结算按钮
    self.log_timing("Click Checkout Button", lambda: self.driver.find_element(By.CSS_SELECTOR, ".cymbal-button-primary:nth-child(1)").click())

    # 8. 查看订单
    self.log_timing("Click Order", lambda: self.driver.find_element(By.CSS_SELECTOR, ".order").click())