import pytest
import random
from sort_module import sort_bus_lines_from_python_list

MOCK_BUSES = [
    {"name": "lineb", "distance": 100, "duration": 40, "frequency": 10},
    {"name": "linea", "distance": 300, "duration": 20, "frequency": 15},
    {"name": "linec", "distance": 50,  "duration": 60, "frequency": 5}
]

# --- ORIGINAL TESTS ---

def test_sort_by_name():
    sorted_buses = sort_bus_lines_from_python_list(MOCK_BUSES, "by_name")
    names = [bus["name"] for bus in sorted_buses]
    assert names == ["linea", "lineb", "linec"]

def test_sort_by_distance():
    sorted_buses = sort_bus_lines_from_python_list(MOCK_BUSES, "by_distance")
    distances = [bus["distance"] for bus in sorted_buses]
    assert distances == [50, 100, 300]

def test_sort_by_duration():
    sorted_buses = sort_bus_lines_from_python_list(MOCK_BUSES, "by_duration")
    durations = [bus["duration"] for bus in sorted_buses]
    assert durations == [20, 40, 60]

def test_empty_list():
    assert sort_bus_lines_from_python_list([], "by_name") == []

def test_invalid_sort_category():
    with pytest.raises(ValueError, match="Unknown sort category"):
        sort_bus_lines_from_python_list(MOCK_BUSES, "by_magic")

def test_missing_dict_keys():
    bad_data = [{"name": "linea", "distance": 100}]
    with pytest.raises(ValueError, match="Missing or malformed numeric fields"):
        sort_bus_lines_from_python_list(bad_data, "by_distance")

def test_invalid_name_type():
    bad_data = [{"name": 123, "distance": 100, "duration": 40, "frequency": 10}]
    # Integers don't have an .encode() method
    with pytest.raises(AttributeError):
        sort_bus_lines_from_python_list(bad_data, "by_name")

def test_duplicate_values():
    duplicates = [
        {"name": "line1", "distance": 100, "duration": 50, "frequency": 10},
        {"name": "line2", "distance": 100, "duration": 40, "frequency": 10},
        {"name": "line3", "distance": 100, "duration": 60, "frequency": 10}
    ]
    sorted_buses = sort_bus_lines_from_python_list(duplicates, "by_distance")
    distances = [bus["distance"] for bus in sorted_buses]
    assert len(sorted_buses) == 3
    assert distances == [100, 100, 100]

def test_c_string_buffer_limit():
    long_name = "a" * 21 # Exactly 21 characters: overwrites the null terminator if unchecked!
    bad_data = [{"name": long_name, "distance": 100, "duration": 40, "frequency": 10}]
    # Our new safety perimeter in sort_module.py strictly prevents >= 21
    with pytest.raises(ValueError, match="exceeds the maximum 20-byte limit"):
        sort_bus_lines_from_python_list(bad_data, "by_name")


# --- NEW ADVANCED EDGE CASES ---

def test_single_item():
    single = [{"name": "onlyone", "distance": 100, "duration": 50, "frequency": 10}]
    sorted_buses = sort_bus_lines_from_python_list(single, "by_distance")
    assert len(sorted_buses) == 1
    assert sorted_buses[0]["name"] == "onlyone"
    assert sorted_buses[0]["distance"] == 100

def test_already_sorted():
    sorted_input = [
        {"name": "line1", "distance": 10, "duration": 20, "frequency": 5},
        {"name": "line2", "distance": 20, "duration": 30, "frequency": 10},
        {"name": "line3", "distance": 30, "duration": 40, "frequency": 15}
    ]
    sorted_buses = sort_bus_lines_from_python_list(sorted_input, "by_distance")
    distances = [bus["distance"] for bus in sorted_buses]
    assert distances == [10, 20, 30]

def test_reverse_sorted():
    reverse_input = [
        {"name": "line1", "distance": 30, "duration": 80, "frequency": 15},
        {"name": "line2", "distance": 20, "duration": 60, "frequency": 10},
        {"name": "line3", "distance": 10, "duration": 40, "frequency": 5}
    ]
    sorted_buses = sort_bus_lines_from_python_list(reverse_input, "by_duration")
    durations = [bus["duration"] for bus in sorted_buses]
    assert durations == [40, 60, 80]

def test_all_identical():
    identical_input = [
        {"name": "clone", "distance": 50, "duration": 50, "frequency": 50}
        for _ in range(5)
    ]
    sorted_buses = sort_bus_lines_from_python_list(identical_input, "by_frequency")
    assert len(sorted_buses) == 5
    for bus in sorted_buses:
        assert bus["distance"] == 50
        assert bus["name"] == "clone"

def test_stress_large_input():
    # Programmatically generate 1000 random bus lines
    stress_input = []
    for i in range(1000):
        stress_input.append({
            "name": f"bus{i}",
            "distance": random.randint(1, 1000),
            "duration": random.randint(10, 100),
            "frequency": random.randint(1, 50)
        })
    
    sorted_buses = sort_bus_lines_from_python_list(stress_input, "by_distance")
    
    # Verify no elements were dropped or duplicated incorrectly
    assert len(sorted_buses) == 1000
    
    # Verify the actual sorting logic held up under stress
    distances = [bus["distance"] for bus in sorted_buses]
    # Check that each element is <= the next element
    assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))

# --- NEW BOUNDARY & CRUEL EDGE CASES ---

def test_exact_20_char_boundary():
    exact_name = "a" * 20
    data = [{"name": exact_name, "distance": 100, "duration": 40, "frequency": 10},
            {"name": "b", "distance": 50, "duration": 20, "frequency": 5}]
    
    # Should successfully process and sort without raising ValueError
    sorted_buses = sort_bus_lines_from_python_list(data, "by_name")
    assert len(sorted_buses) == 2
    assert sorted_buses[0]["name"] == exact_name
    assert sorted_buses[1]["name"] == "b"

def test_empty_name_string():
    data = [{"name": "", "distance": 100, "duration": 40, "frequency": 10},
            {"name": "a", "distance": 50, "duration": 20, "frequency": 5}]
    
    # Empty string should evaluate as less than "a" and pass C-string conversions perfectly
    sorted_buses = sort_bus_lines_from_python_list(data, "by_name")
    assert sorted_buses[0]["name"] == ""

def test_exact_numeric_boundaries():
    data = [
        {"name": "limit_low", "distance": 0, "duration": 10, "frequency": 1},
        {"name": "limit_high", "distance": 1000, "duration": 100, "frequency": 50},
        {"name": "mid", "distance": 500, "duration": 50, "frequency": 25}
    ]
    
    # Duration exactly at 10 and 100
    sorted_buses = sort_bus_lines_from_python_list(data, "by_duration")
    durations = [bus["duration"] for bus in sorted_buses]
    assert durations == [10, 50, 100]

def test_negative_numeric_values():
    data = [
        {"name": "neg_dist", "distance": -500, "duration": 40, "frequency": -10},
        {"name": "pos_dist", "distance": 100, "duration": 20, "frequency": 5},
        {"name": "zero_dist", "distance": 0, "duration": 60, "frequency": 0}
    ]
    
    # Sorting mathematically with negatives
    sorted_buses = sort_bus_lines_from_python_list(data, "by_distance")
    distances = [bus["distance"] for bus in sorted_buses]
    assert distances == [-500, 0, 100]
    
    sorted_buses_freq = sort_bus_lines_from_python_list(data, "by_frequency")
    freqs = [bus["frequency"] for bus in sorted_buses_freq]
    assert freqs == [-10, 0, 5]

# --- ADVANCED C-MEMORY & FFI TESTS ---
from sort_module import BusLine, lib, DISTANCE
import ctypes

def test_canary_memory_bounds():
    # Allocate an array with 5 elements (2 canaries + 3 items)
    BusLineArray = BusLine * 5
    c_array = BusLineArray()
    
    # Setup Canary 1
    c_array[0].name = b"CANARY_START"
    c_array[0].distance = -999
    
    # Setup Real Array (unsorted)
    c_array[1].name = b"Charlie"
    c_array[1].distance = 30
    c_array[2].name = b"Alice"
    c_array[2].distance = 10
    c_array[3].name = b"Bob"
    c_array[3].distance = 20
    
    # Setup Canary 2
    c_array[4].name = b"CANARY_END"
    c_array[4].distance = 999
    
    # Extract strict bounds for the inner slice [1 to 3]
    start = ctypes.cast(ctypes.byref(c_array[1]), ctypes.POINTER(BusLine))
    end = ctypes.cast(ctypes.byref(c_array[3]), ctypes.POINTER(BusLine))
    
    # Execute C sort directly on the inner slice
    lib.bus_quick_sort(start, end, DISTANCE)
    
    # Assert Sorting Succeeded
    assert c_array[1].distance == 10
    assert c_array[2].distance == 20
    assert c_array[3].distance == 30
    
    # Assert Canaries are 100% untouched
    assert c_array[0].name == b"CANARY_START"
    assert c_array[0].distance == -999
    assert c_array[4].name == b"CANARY_END"
    assert c_array[4].distance == 999

def test_ctypes_pointer_integrity():
    N = 10
    BusLineArray = BusLine * N
    c_array = BusLineArray()
    
    start_ptr = ctypes.cast(c_array, ctypes.POINTER(BusLine))
    end_ptr = ctypes.cast(ctypes.byref(c_array[N - 1]), ctypes.POINTER(BusLine))
    
    # Extract absolute memory addresses
    start_address = ctypes.addressof(start_ptr.contents)
    end_address = ctypes.addressof(end_ptr.contents)
    
    # Assert pointer difference perfectly matches C struct stride
    expected_difference = ctypes.sizeof(BusLine) * (N - 1)
    actual_difference = end_address - start_address
    
    assert actual_difference == expected_difference
