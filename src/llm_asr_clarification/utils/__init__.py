import re

def extract_timestamps(line):
    # Match either a float ("15.87") or a plain integer ("15") — handles both
    # old GT files (integer timestamps) and new ones (float timestamps).
    numbers = re.findall(r'\d+(?:\.\d+)?', line)

    # Return floats so sub-second precision is preserved end-to-end
    start_time = float(numbers[0])
    end_time = float(numbers[1])

    return start_time, end_time