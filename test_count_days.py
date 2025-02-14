#!/usr/bin/env python3
from dateutil import parser
import os

def count_wednesdays(file_path):
    """
    Reads a file line by line, attempts to parse each line as a date using dateutil.parser,
    and counts how many of the parsed dates fall on a Wednesday (weekday == 2).
    
    Debug statements print the original date string, the parsed datetime, and its weekday.
    """
    total_wednesdays = 0

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        for line in file:
            date_str = line.strip()
            if not date_str:
                continue

            try:
                # Parse the date string using fuzzy parsing
                dt = parser.parse(date_str, fuzzy=True)
                # Debug print: show the original string, parsed datetime, and weekday number
                print(f"Parsed: '{date_str}' -> {dt} (weekday: {dt.weekday()})")
                if dt.weekday() == 2:  # In Python's datetime.weekday(), Monday is 0 and Wednesday is 2
                    total_wednesdays += 1
            except Exception as e:
                print(f"Error parsing '{date_str}': {e}")
    
    return total_wednesdays

if __name__ == '__main__':
    # Set the path to your dates.txt file. Adjust this path as necessary.
    file_path = '/Users/mish/Documents/llm/data/dates.txt'
    try:
        count = count_wednesdays(file_path)
        print(f"\nCounted {count} occurrences of Wednesday.")
    except Exception as e:
        print(f"An error occurred: {e}")
