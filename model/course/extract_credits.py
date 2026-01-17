#! /usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import sys
import time

def parse_courses(contents):
    parser = BeautifulSoup(contents)
    entries = parser.find_all("h3")
    courses = []
    for entry in entries:
        for child in entry.children:
            if child.name == "a":
                if "BUS" in child.text:
                    try:
                        courses.append((child.text.split()[1], entry.text.split('(')[1].split(')')[0]))
                    except:
                        pass # dont know dont care(likely graduate courses)
    return courses

path = None
try:
    start = int(sys.argv[1])
    end = 2026
except:
    path = sys.argv[1]


if path:
    with open(path) as f:
        print(parse_courses(f))
else:
    with open("credits.py", "w") as credits:
        credits.write("# Generated using command: "+" ".join(sys.argv)+"\n")
        credits.write("courses = {\n")
        for year in range(start, end+1):
            for term_index,term in ((1,'spring'),(4,'summer'), (7, 'fall')):
                course_term = f"1{str(year)[2:]}{term_index}"
                time.sleep(1)
                try:
                    url = f"https://www.sfu.ca/students/calendar/{year}/{term}/courses/bus.html"
                    print(f"downloading {url}")
                    req = requests.get(url)
                    if req.status_code == 200:
                        courses = parse_courses(req.text)
                        for course in courses:
                            credits.write(f"    ('{course[0]}',{course_term}):{course[1]},\n")
                except:
                    pass
        credits.write("}")
