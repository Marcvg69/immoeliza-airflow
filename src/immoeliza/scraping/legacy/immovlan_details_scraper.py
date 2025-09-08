import glob
import os
import csv
import time
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from immoeliza.scraping.common import logger

class ImmovlanDetailsScraper:
    """
    ImmovlanDetailsScraper is a class designed to extract detailed real estate property information from Immovlan URLs listed in a consolidated CSV file. It uses Selenium WebDriver to navigate property detail pages and BeautifulSoup to parse and extract relevant data fields, saving the results to a timestamped CSV file.
    """

    def __init__(self, output_dir: str = "output", headless: bool = True, limit: int = -1):
        """
        Initializes the scraper with specified output directory, headless mode, and result limit.
        """

        self.output_dir = output_dir
        """Directory where output files will be saved."""
        
        self.limit = limit
        """Maximum number of items to process. Use -1 for no limit."""

        self.headless = (os.getenv('IMMO_HEADLESS','true').lower()!='false') if headless is True else headless
        """Flag to indicate if the browser should run in headless mode."""
        
        self.csv_file = self._get_latest_consolidated_csv()
        """ Path to the latest consolidated CSV file containing property URLs."""

        self.driver = self._init_driver()
        """Selenium WebDriver instance for navigating property detail pages."""

        self.output_file = self._generate_output_file_path()
        """Path to the output CSV file for storing extracted property details."""

    def _init_driver(self):
        """
        Initializes and configures a Chrome WebDriver instance with custom options.

        Sets up the Chrome WebDriver with a specified user-agent, disables extensions and GPU usage,
        sets the window size, and applies additional options for running in environments such as Docker.
        If the 'headless' attribute is True, the browser will run in headless mode.

        Returns:
            selenium.webdriver.Chrome: A configured instance of Chrome WebDriver.
        """
        options = Options()
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        if self.headless:
            options.add_argument("--headless=new")        
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def _generate_output_file_path(self):
        """
        Generates a unique output file path for storing real estate details as a CSV file.

        The method creates a timestamped directory within the specified output directory,
        ensuring the directory exists. It then constructs a CSV file path within this directory,
        using the same timestamp in the filename.

        Returns:
            str: The full path to the timestamped CSV output file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        dir_path = os.path.join(self.output_dir, f"_real_estate_details_{timestamp}")
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, f"_real_estate_details_{timestamp}.csv")

    def _get_latest_consolidated_csv(self):
        """
        Finds and returns the most recently modified consolidated CSV file in the output directory.

        The method searches for files matching the pattern '_consolidated_towns_urls_*/_consolidated_towns_urls_*.csv'
        within the specified output directory. If no matching files are found, a FileNotFoundError is raised.

        Returns:
            str: The path to the most recently modified consolidated CSV file.

        Raises:
            FileNotFoundError: If no consolidated CSV files are found in the output directory.
        """
        pattern = os.path.join(self.output_dir, "_consolidated_towns_urls_*/_consolidated_towns_urls_*.csv")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError("No consolidated CSV file found.")
        return max(files, key=os.path.getmtime)

    def scrape_and_save_properties(self):
        """
        Extracts detailed property information from URLs listed in a CSV file and writes the results to an output CSV file.
        This method reads a CSV file specified by `self.csv_file`, optionally limits the number of rows processed,
        and iterates through each property URL. For each URL, it navigates to the page using Selenium, parses the HTML
        with BeautifulSoup, and extracts various property details such as price, address, number of bedrooms, surface areas,
        and energy performance data. The extracted data is written to a new CSV file specified by `self.output_file`.
        """
        logger.info("🔍 Reading from: %s", self.csv_file)
        df = pd.read_csv(self.csv_file)
        if self.limit != -1:
            df = df.head(self.limit)

        fieldnames = [
            "town", "page", "url", "property_type", "price", "address", "postal_code", "city",
            "bedrooms", "bedroom1_surface", "bedroom2_surface", "bathrooms", "toilets",
            "surface_livable", "terrace", "terrace_surface", "terrace_orientation", "floor", "year_built",
            "condition", "kitchen_equipment", "cellar", "glazing_type", "elevator", "entry_phone",
            "epc_score", "epc_total", "epc_valid_until"
        ]

        # Open the output CSV file for writing, using UTF-8 encoding and disabling newline translation
        with open(self.output_file, mode="w", newline="", encoding="utf-8") as f:
            
            # Create a CSV writer that writes dictionaries with the specified fieldnames
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write the CSV header (column names)
            writer.writeheader()

            # Loop over each row in the DataFrame, without using the index
            for i, row in enumerate(df.itertuples(index=False), start=1):
                try:
                    # Use the Selenium driver to load the URL of the property
                    self.driver.get(row.url)
                    
                    # Wait for 2 seconds to allow the page to fully load (basic throttling)
                    time.sleep(2)

                    # Parse the page source using BeautifulSoup
                    soup = BeautifulSoup(self.driver.page_source, "html.parser")

                    # Define a helper function to extract values associated with specific <h4> labels
                    def get_label_value(label):
                        element = soup.find("h4", string=label)
                        if element and element.find_next_sibling("p"):
                            return element.find_next_sibling("p").text.strip()
                        return None

                    # Build a dictionary with the extracted details for each property
                    details = {
                        "town": getattr(row, "town", ""),
                        "page": getattr(row, "page", ""),
                        "url": row.url,
                        "property_type": soup.select_one(".detail__header_title_main").text.strip().split()[0] if soup.select_one(".detail__header_title_main") else None,
                        "price": soup.select_one(".detail__header_price_data").text.strip() if soup.select_one(".detail__header_price_data") else None,
                        "address": soup.select_one(".detail__header_address").text.strip() if soup.select_one(".detail__header_address") else None,
                        "city": None,
                        "bedrooms": get_label_value("Number of bedrooms"),
                        "bedroom1_surface": get_label_value("Surface bedroom 1"),
                        "bedroom2_surface": get_label_value("Surface bedroom 2"),
                        "bathrooms": get_label_value("Number of bathrooms"),
                        "toilets": get_label_value("Number of toilets"),
                        "surface_livable": get_label_value("Livable surface"),
                        "terrace": get_label_value("Terrace"),
                        "terrace_surface": get_label_value("Surface terrace"),
                        "terrace_orientation": get_label_value("Terrace orientation"),
                        "floor": get_label_value("Floor of appartment"),
                        "year_built": get_label_value("Build Year"),
                        "condition": get_label_value("State of the property"),
                        "kitchen_equipment": get_label_value("Kitchen equipment"),
                        "cellar": get_label_value("Cellar"),
                        "glazing_type": get_label_value("Type of glazing"),
                        "elevator": get_label_value("Elevator"),
                        "entry_phone": get_label_value("Entry phone"),
                        "epc_score": get_label_value("Specific primary energy consumption"),
                        "epc_total": get_label_value("Yearly total primary energy consumption"),
                        "epc_valid_until": get_label_value("Validity date EPC/PEB")
                    }

                    # Try to extract the postal code and city name from the address string
                    address_parts = details["address"].split() if details["address"] else []
                    if len(address_parts) >= 2:
                        try:
                            details["postal_code"] = next(part for part in address_parts if part.isdigit())
                            details["city"] = address_parts[-1]
                        except StopIteration:
                            pass

                    # Write the extracted data as a new row in the CSV
                    writer.writerow(details)

                     # Log successful extraction for this property
                    logger.info("✅ [%d/%d] Extracted: %s", i, len(df), row.url)
                    
                except Exception as e:
                    # Log any errors encountered while processing this property
                    logger.warning("⚠️ Error extracting from %s: %s", row.url, e)

        # After finishing the loop, log that all data has been saved successfully
        logger.info("💾 Saved detailed data to: %s", self.output_file)

    def close(self):
        """
        Closes the web driver and releases all associated resources.
        """
        self.driver.quit()
