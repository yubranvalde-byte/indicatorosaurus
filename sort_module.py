import ctypes
import os

# --- 1. Data Structure Definitions ---

# Define the C enum SortType as Python constants
DISTANCE = 0
DURATION = 1
FREQUENCY = 2

class BusLine(ctypes.Structure):
    """
    ctypes mapping for the C BusLine struct:
    typedef struct BusLine {
        char name[21];
        int distance, duration, frequency;
    } BusLine;
    """
    _fields_ = [
        ("name", ctypes.c_char * 21),
        ("distance", ctypes.c_int),
        ("duration", ctypes.c_int),
        ("frequency", ctypes.c_int)
    ]

# --- 2. Library Loading & Function Signatures ---

# Determine the absolute path to the compiled shared library
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsortbuslines.so")

try:
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"Error loading {lib_path}. Did you compile it?")
    raise e

# Setup bus_bubble_sort signature
# void bus_bubble_sort (BusLine *start, BusLine *end);
lib.bus_bubble_sort.argtypes = [ctypes.POINTER(BusLine), ctypes.POINTER(BusLine)]
lib.bus_bubble_sort.restype = None

# Setup bus_quick_sort signature
# void bus_quick_sort (BusLine *start, BusLine *end, SortType sort_type);
lib.bus_quick_sort.argtypes = [ctypes.POINTER(BusLine), ctypes.POINTER(BusLine), ctypes.c_int]
lib.bus_quick_sort.restype = None

# Setup visualize_bus_bubble_sort signature
lib.visualize_bus_bubble_sort.argtypes = [ctypes.POINTER(BusLine), ctypes.POINTER(BusLine)]
lib.visualize_bus_bubble_sort.restype = None

# Setup visualize_bus_quick_sort signature
lib.visualize_bus_quick_sort.argtypes = [ctypes.POINTER(BusLine), ctypes.POINTER(BusLine), ctypes.c_int]
lib.visualize_bus_quick_sort.restype = None

# --- 3. Pythonic Helper Function ---

def sort_bus_lines_from_python_list(buses_list, sort_category="by_name"):
    """
    Takes a list of dictionaries, sorts them using the C backend, and returns a new sorted list.
    sort_category can be: "by_name", "by_distance", "by_duration", "by_frequency"
    """
    if not buses_list:
        return []
        
    # Security Fix: Prevent native C-stack Segmentation Faults on degenerate O(N) recursive QuickSort inputs
    if len(buses_list) > 2500:
        raise ValueError("Array size exceeds maximum safe recursion depth for native execution.")
    
    # 1. Create the ctypes array
    num_buses = len(buses_list)
    BusLineArray = BusLine * num_buses
    c_array = BusLineArray()
    
    # Populate the array
    for i, bus_dict in enumerate(buses_list):
        # Encode string to bytes for the C char array
        encoded_name = bus_dict["name"].encode('utf-8')
        
        # Security Fix: Enforce max 20 bytes to guarantee a null terminator '\0' at index 20
        if len(encoded_name) > 20:
            raise ValueError(f"Bus name '{bus_dict['name']}' exceeds the maximum 20-byte limit.")
            
        c_array[i].name = encoded_name
        
        # Type Safety Fix: Explicitly cast to int to prevent ctypes from raising cryptic TypeErrors
        try:
            c_array[i].distance = int(bus_dict["distance"])
            c_array[i].duration = int(bus_dict["duration"])
            c_array[i].frequency = int(bus_dict["frequency"])
        except (ValueError, TypeError, KeyError, OverflowError) as e:
            # Safely extract name for error message without triggering another KeyError
            bus_name = bus_dict.get('name', 'UNKNOWN_BUS')
            raise ValueError(f"Missing or malformed numeric fields for bus '{bus_name}'.") from e
        
    # 2. Calculate start and end pointers using precise memory offsets
    # start_ptr points to the beginning of the array
    start_ptr = ctypes.cast(c_array, ctypes.POINTER(BusLine))
    
    # end_ptr points to the LAST element in the array (index: num_buses - 1)
    # Refactored for safer, cleaner Pythonic pointer extraction:
    end_ptr = ctypes.cast(ctypes.byref(c_array[num_buses - 1]), ctypes.POINTER(BusLine))
    
    # 3. Call the appropriate C sorting function
    if sort_category == "by_name":
        lib.bus_bubble_sort(start_ptr, end_ptr)
    elif sort_category == "by_distance":
        lib.bus_quick_sort(start_ptr, end_ptr, DISTANCE)
    elif sort_category == "by_duration":
        lib.bus_quick_sort(start_ptr, end_ptr, DURATION)
    elif sort_category == "by_frequency":
        lib.bus_quick_sort(start_ptr, end_ptr, FREQUENCY)
    else:
        raise ValueError(f"Unknown sort category: {sort_category}")
        
    # 4. Reconstruct the sorted python list
    sorted_buses = []
    for i in range(num_buses):
        sorted_buses.append({
            # Decode bytes back to Python string
            "name": c_array[i].name.decode('utf-8'),
            "distance": c_array[i].distance,
            "duration": c_array[i].duration,
            "frequency": c_array[i].frequency
        })
        
    return sorted_buses
