import json
import os
from playwright.sync_api import sync_playwright
import random
import re
import urllib.parse

# Import list of destinations
with open('data/city_list.txt') as f:
    cities = [line.strip() for line in f]

# JSON file output path
json_file_path = 'data/hotel_info.json'

# Initialize a list to store hotels data
hotels_data = []

with sync_playwright() as p:
    
    # Create browser instance
    browser = p.chromium.launch(headless=False)
    
    page = browser.new_page()

    for city in cities:
        print(f"\n--- City: {city} ---")
        
        search_url = "https://www.booking.com/searchresults.fr.html?ss=" + urllib.parse.quote(city)
        page.goto(search_url, timeout=60000)

        # Handle cookies pop up
        try:
            page.locator("button:has-text('Accepter')").click(timeout=5000)
            print("Cookies accepted")
        except:
            print("No popup cookies")

        # Wait for "property-card" to appear
        page.wait_for_selector("div[data-testid='property-card']", timeout=20000)

        # Retrieve infos
        hotels = page.locator("div[data-testid='property-card']")
        count = hotels.count()
        print(f"{count} hotels found")

        # Extract data for the first 5 hotels
        for i in range(min(5, count)):
            card = hotels.nth(i)

            name = card.locator("div[data-testid='title']").inner_text()
            link = card.locator("a[data-testid='title-link']").get_attribute("href")
            
            if link.startswith("/"):
                link = "https://www.booking.com" + link

            print(name)
            print(link)
            
            hotel_page = browser.new_page()
            hotel_page.goto(link, timeout=60000)
            hotel_page.wait_for_load_state("networkidle")

            # Retrieve rating
            try:      
                rating_text = hotel_page.locator("div[data-testid='review-score-right-component'] div.f63b14ab7a.dff2e52086").inner_text()
                rating = float(rating_text.replace(',', '.'))  # Convertir directement en float
            except Exception as e:
                print(f"Error retrieving rating: {e}")
                rating = None
                
            # Retrieve number of reviews
            try:
                reviews_text = hotel_page.locator(
                    "div[data-testid='review-score-right-component'] div.fff1944c52"
                ).inner_text()
                reviews_count = int(''.join(re.findall(r'\d+', reviews_text)))
            except Exception as e:
                print(f"Error retrieving reviews count: {e}")
                reviews_count = None

            # Retrieve description
            try:
                description = hotel_page.locator("p[data-testid='property-description']").inner_text()
            except Exception as e:
                print(f"Error retrieving description: {e}")
                description = None
            
            # Retrieve geographical coordinates (latitude, longitude)
            try:
                latlng = hotel_page.locator("a#map_trigger_header_pin").get_attribute("data-atlas-latlng")
                if latlng:
                    lat_str, lon_str = latlng.replace('\u200b', '').split(',')
                    lat = float(lat_str)
                    lon = float(lon_str)
                else:
                    print("Error retrieving latitude and longitude")
                    lat, lon = None, None
            except Exception as e:
                print("Error retrieving latitude and longitude")
                lat, lon = None, None
        
            # Display infos
            print(f"Rating: {rating}")
            print(f"Reviews: {reviews_count}")
            print(f"Description: {description}")
            print(f"Latitude: {lat}, Longitude: {lon}")
            print("-" * 40)
        
            # Store infos in a dictionary
            hotel_info = {
                "city": city,
                "hotel_name": name,
                "hotel_url": link,
                "hotel_latitude": lat,
                "hotel_longitude": lon,
                "hotel_rating": rating,
                "hotel_reviews": reviews_count,
                "hotel_description": description
            }

            # Update the list with additional hotel information
            hotels_data.append(hotel_info)
            
            hotel_page.close()

            # Pause between each hotel search
            page.wait_for_timeout(random.randint(1000, 2000))
            
        # Pause between each city search
        page.wait_for_timeout(random.randint(3000, 5000))
        
    # Check the existence of the JSON file. If yes, display a message indicating that it will be replaced
    if os.path.exists(json_file_path):
        print(f"The {json_file_path} file exists and will be replaced.")
        
    # Export data in a JSON file
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(hotels_data, json_file, ensure_ascii=False, indent=4)
        
    browser.close()
