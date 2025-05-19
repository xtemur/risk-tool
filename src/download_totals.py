# import os
import io
import os
import re
from datetime import date, timedelta

import pandas as pd
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
      - current_page: current page number (default is 1)
      - total_pages: total number of pages found (default is 1)
    """
    lines = csv_text.strip().splitlines()
    current_page = 1
    total_pages = 1  # Default if no pagination marker exists
    if lines and re.match(r"Page\s+\d+/\d+", lines[-1].strip()):
        # Remove the pagination marker from the lines and extract total pages
        pagination_line = lines.pop().strip()
        # Use a regex that captures both current page (group 1) and total pages (group 2)
        m_pagination = re.search(r"Page\s+(\d+)\s*/\s*(\d+)", pagination_line)
        if m_pagination:
            current_page = int(m_pagination.group(1))  # e.g., the '5' from "Page 5/10"
            total_pages = int(m_pagination.group(2))  # e.g., the '10' from "Page 5/10"
        # If m_pagination doesn't match (though unlikely if re.match passed),
        # current_page and total_pages will retain their default values of 1.
    return lines, current_page, total_pages


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


def fetch_pages(base_data):
    page_num = 0
    header = ""
    all_data_lines = []
    while True:
        page_num += 1

        base_data["page"] = str(page_num)
        csv_text = fetch_csv_page(base_data)
        if not csv_text:
            print(f"Skipping page {page_num} due to fetch error.")
            continue
        page_lines, current_page, total_pages = parse_csv_response(csv_text)
        if not page_lines:
            continue
        if header == "":
            header = page_lines[0]
        data_lines = page_lines[1:] if len(page_lines) > 1 else []
        all_data_lines.extend(data_lines)

        if current_page >= total_pages:
            break
    return header, all_data_lines


def get_account_df(token):
    base_data = {"action": "accounts", "token": token}
    data = fetch_pages(base_data)
    if data is None:
        print("Failed to download data.")
        return
    header, data_lines = data
    # parse it to a pandas dataframe
    df = pd.read_csv(io.StringIO("\n".join([header] + data_lines)))

    return df


def get_totals_by_date_per_account(token, overall_start, overall_end, account_id_list):
    reports_data = {"action": "report", "type": "totalsByDate", "token": token}

    for account_id in account_id_list:
        base_data = reports_data.copy()
        base_data["accountId"] = account_id
        for month_start in month_date_range(overall_start, overall_end):
            month_end = get_last_day_of_month(month_start)
            start_date_str = month_start.strftime("%Y-%m-%d")
            end_date_str = month_end.strftime("%Y-%m-%d")
            base_data["startDate"] = start_date_str
            base_data["endDate"] = end_date_str

            print(f"Downloading data for {start_date_str} to {end_date_str} ...")
            header, data_lines = fetch_pages(base_data)

            if header is None:
                print(
                    f"Failed to download data for {start_date_str} to {end_date_str}."
                )
                continue

            # Merge header and all data lines into one CSV string
            csv_combined = "\n".join([header] + data_lines)
            # Save the CSV to a file. The file name includes the year and month.
            # make dir if not exist
            if not os.path.exists(f"data/raw/totals_by_date/{account_id}"):
                os.makedirs(f"data/raw/totals_by_date/{account_id}")
            file_name = f"data/raw/totals_by_date/{account_id}/tbd_{account_id}_{month_start.strftime('%Y_%m')}.csv"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(csv_combined)
            print(f"Data for {start_date_str} to {end_date_str} saved to {file_name}.")


def get_fills_per_account(token, overall_start, overall_end, account_id_list):
    fills_data = {"action": "fills", "type": "totalsByDate", "token": token}

    for account_id in account_id_list:
        base_data = reports_data.copy()
        base_data["accountId"] = account_id
        for month_start in month_date_range(overall_start, overall_end):
            month_end = get_last_day_of_month(month_start)
            start_date_str = month_start.strftime("%Y-%m-%d")
            end_date_str = month_end.strftime("%Y-%m-%d")
            base_data["startDate"] = start_date_str
            base_data["endDate"] = end_date_str

            print(f"Downloading data for {start_date_str} to {end_date_str} ...")
            header, data_lines = fetch_pages(base_data)

            if header is None:
                print(
                    f"Failed to download data for {start_date_str} to {end_date_str}."
                )
                continue

            # Merge header and all data lines into one CSV string
            csv_combined = "\n".join([header] + data_lines)
            # Save the CSV to a file. The file name includes the year and month.
            # make dir if not exist
            if not os.path.exists(f"data/raw/fills/{account_id}"):
                os.makedirs(f"data/raw/fills/{account_id}")
            file_name = f"data/raw/fills/{account_id}/fills_{account_id}_{month_start.strftime('%Y_%m')}.csv"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(csv_combined)
            print(f"Data for {start_date_str} to {end_date_str} saved to {file_name}.")


def main():

    overall_start = date(2022, 4, 1)
    overall_end = date(2025, 2, 28)

    # Your API token and IDs
    token = "5a764d70b5640a56243e131cb52183fd:2523"

    account_df = get_account_df(token)

    all_accounts = account_df["Account Id"].tolist()
    print(all_accounts)
    get_totals_by_date_per_account(token, overall_start, overall_end, all_accounts)

    print(accounts_id)

    # get_totals_by_date(token, overall_start, overall_end, accounts_id)

    # Base form data that is shared by all requests
    # If you want to use groupId instead, just replace accountId with groupId and remove accountId.

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


if __name__ == "__main__":
    main()
