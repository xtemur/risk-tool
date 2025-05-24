# import os
import re
from datetime import date, timedelta

import requests

# Replace this URL with your API endpoint
API_URL = "https://neo.propreports.com/api.php"

# Common headers for sending the data as x-www-form-urlencoded
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


def fetch_csv_page(form_data):
    """
    Send a POST request with the provided form data.
    Returns the CSV text if status 200, otherwise None.
    """
    response = requests.post(API_URL, data=form_data, headers=HEADERS)
    print(f"Fetching page {form_data.get('page')} with data: {form_data}")
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error: HTTP {response.status_code} for page {form_data.get('page')}")
        return None


def parse_csv_response(csv_text):
    """
    Splits the CSV text into lines, and checks if the last line is a pagination marker.

    Returns:
      - lines: list of CSV lines (without the pagination marker if it existed)
      - total_pages: total number of pages found (default is 1)
    """
    lines = csv_text.strip().splitlines()
    total_pages = 1  # Default if no pagination marker exists
    if lines and re.match(r"Page\s+\d+/\d+", lines[-1].strip()):
        # Remove the pagination marker from the lines and extract total pages
        pagination_line = lines.pop().strip()
        m = re.search(r"/\s*(\d+)", pagination_line)
        if m:
            total_pages = int(m.group(1))
    return lines, total_pages


def month_date_range(start_date, end_date):
    """
    Generator that yields the first day of each month between start_date and end_date.
    The end_date is exclusive.
    """
    current = start_date.replace(day=1)
    while current < end_date:
        yield current
        # Move to the next month
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        current = next_month


def get_last_day_of_month(dt):
    """
    Returns the last day of the month for the given date dt.
    """
    next_month = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
    return next_month - timedelta(days=1)


def fetch_month_pages(start_date_str, end_date_str, base_data):
    """
    For the given date range (for one month) and base form data, request and merge all pages.

    Returns:
       header: the CSV header line from the first page.
       all_data_lines: all CSV data rows combined from the multiple pages.
    """
    all_data_lines = []
    # Copy base_data to avoid modifying original
    form_data = base_data.copy()
    form_data["page"] = "1"

    # Fetch the first page
    csv_text = fetch_csv_page(form_data)
    if not csv_text:
        return None, None

    lines, total_pages = parse_csv_response(csv_text)
    if not lines:
        print(f"No CSV content returned for {start_date_str} to {end_date_str}")
        return None, None

    # The first line is assumed to be the header
    header = lines[0]
    # For page 1, the rest are data
    data_lines = lines[1:]
    all_data_lines.extend(data_lines)

    # If there is pagination (more than one page), fetch the rest
    if total_pages > 1:
        for page in range(2, total_pages + 1):
            form_data["page"] = str(page)
            csv_text = fetch_csv_page(form_data)
            if not csv_text:
                print(f"Skipping page {page} due to fetch error.")
                continue
            page_lines, _ = parse_csv_response(csv_text)
            if not page_lines:
                continue
            # Remove the header row from each page so that it does not get repeated
            # (Assumes that the header is on every page)
            page_data = page_lines[1:] if len(page_lines) > 1 else []
            all_data_lines.extend(page_data)

    return header, all_data_lines


def fetch_pages(base_data):
    """
    For the given date range (for one month) and base form data, request and merge all pages.

    Returns:
       header: the CSV header line from the first page.
       all_data_lines: all CSV data rows combined from the multiple pages.
    """
    all_data_lines = []
    # Copy base_data to avoid modifying original
    form_data = base_data.copy()
    form_data["page"] = "1"

    # Fetch the first page
    csv_text = fetch_csv_page(form_data)
    if not csv_text:
        return None, None

    lines, total_pages = parse_csv_response(csv_text)
    if not lines:
        print(
            f"No CSV content returned for {form_data['startDate']} to {form_data['endDate']}"
        )
        return None, None

    # The first line is assumed to be the header
    header = lines[0]
    # For page 1, the rest are data
    data_lines = lines[1:]
    all_data_lines.extend(data_lines)

    # If there is pagination (more than one page), fetch the rest
    if total_pages > 1:
        for page in range(2, total_pages + 1):
            print("Downloading page #{}/ {}", page, total_pages)
            form_data["page"] = str(page)
            csv_text = fetch_csv_page(form_data)
            if not csv_text:
                print(f"Skipping page {page} due to fetch error.")
                continue
            page_lines, _ = parse_csv_response(csv_text)
            if not page_lines:
                continue
            # Remove the header row from each page so that it does not get repeated
            # (Assumes that the header is on every page)
            page_data = page_lines[1:] if len(page_lines) > 1 else []
            all_data_lines.extend(page_data)

    return header, all_data_lines


def main():
    # Your API token and IDs
    token = "a3b113e76588246dac5b528a1f82a93d:2523"

    # Base form data that is shared by all requests
    # If you want to use groupId instead, just replace accountId with groupId and remove accountId.

    overall_start = date(2023, 4, 1)
    overall_end = date(2025, 2, 28)

    # base_data = {
    #     "action": "fills",
    #     "token": token,
    #     "startDate": overall_start.strftime("%Y-%m-%d"),
    #     "endDate": overall_end.strftime("%Y-%m-%d"),
    #     # "accountId": accountId,
    #     "groupId": -4,
    # }

    # positions_data = {
    #     "type": "all",
    #     "action": "positions",
    #     "token": token,
    #     "startDate": overall_start.strftime("%Y-%m-%d"),
    #     "endDate": overall_end.strftime("%Y-%m-%d"),
    #     "groupId": -4,
    # }

    reports_data = {
        "action": "report",
        "type": "totalsByDate",
        # "groupId": -4,
        "accountId": 4004,
        "startDate": overall_start.strftime("%Y-%m-%d"),
        "endDate": overall_end.strftime("%Y-%m-%d"),
        "token": token,
    }

    header, data_lines = fetch_pages(reports_data)
    if header is None:
        print("Failed to download data.")
        return

    file_name = "data/raw/totalsByDate_new.csv"

    with open(file_name, "w", encoding="utf-8") as f:
        f.write("\n".join([header] + data_lines))
    print(f"Data for {overall_start} to {overall_end} saved to {file_name}.")

    # header, data_lines = fetch_pages(positions_data)
    # if header is None:
    #     print("Failed to download data.")
    #     return

    # file_name = f"data/raw/positions/positions_{overall_end.strftime('%Y_%m')}.csv"

    # with open(file_name, "w", encoding="utf-8") as f:
    #     f.write("\n".join([header] + data_lines))
    # print(f"Data for {overall_start} to {overall_end} saved to {file_name}.")

    # Overall date range

    # Loop over each month in the date range

    # for month_start in month_date_range(overall_start, overall_end):
    #     month_end = get_last_day_of_month(month_start)
    #     start_date_str = month_start.strftime("%Y-%m-%d")
    #     end_date_str = month_end.strftime("%Y-%m-%d")

    #     print(f"Downloading data for {start_date_str} to {end_date_str} ...")
    #     header, data_lines = fetch_month_pages(start_date_str, end_date_str, base_data)

    #     if header is None:
    #         print(f"Failed to download data for {start_date_str} to {end_date_str}.")
    #         continue

    #     # Merge header and all data lines into one CSV string
    #     csv_combined = "\n".join([header] + data_lines)
    #     # Save the CSV to a file. The file name includes the year and month.
    #     file_name = f"data/raw/fills/fills_{month_start.strftime('%Y_%m')}.csv"
    #     with open(file_name, "w", encoding="utf-8") as f:
    #         f.write(csv_combined)
    #     print(f"Data for {start_date_str} to {end_date_str} saved to {file_name}.")


if __name__ == "__main__":
    main()
