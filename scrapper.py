from selenium import webdriver
from selenium.webdriver.common.by import By
import bs4
import time


def scrape_links(soup):
    gs_title_links = soup.find_all('a', class_='gs-title')
    return gs_title_links


def print_links(links, view_links):
    for link in links:
        if view_links == 'y':
            href = link.get('href')
            print(f"Link: {href}")
        text = link.get_text()
        print(f"Title: {text}\n")


def main():
    cwe_search = input("Enter CWE search term: ")
    owasp_search = input("Enter OWASP search term (if blank it will take CWE search input): ")
    view_links = input("Do you want to view links? (y/n): ")

    if not owasp_search:
        owasp_search = cwe_search

    cwe_link = f"https://cwe.mitre.org/find/index.html"
    owasp_link = f"https://owasp.org/search/?searchString={owasp_search}"

    driver = webdriver.Firefox()
    try:
        # Scrap headers from CWE
        driver.get(cwe_link)
        search = driver.find_element(By.XPATH, '//*[@id="gsc-i-id1"]')
        search.send_keys(cwe_search)
        btn = driver.find_element(By.XPATH, '//*[@id="___gcse_0"]/div/div/form/table/tbody/tr/td[2]/button')
        btn.click()

        # Introduce sleep time to allow the page to load
        time.sleep(3)

        soup = bs4.BeautifulSoup(driver.page_source, "html.parser")
        print("############################# CWE Data #############################")
        cwe_links = scrape_links(soup)
        print_links(cwe_links, view_links)

        # Introduce sleep time to Wait fetch data and print it
        time.sleep(5)
        # Scrap headers from OWASP
        driver.get(owasp_link)

        # Introduce sleep time for the OWASP page to load
        time.sleep(3)

        soup = bs4.BeautifulSoup(driver.page_source, "html.parser")
        print("############################# OWASP Data #############################")
        owasp_links = scrape_links(soup)
        print_links(owasp_links, view_links)

    finally:
        driver.quit()

    return True


if __name__ == "__main__":
    main()