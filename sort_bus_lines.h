#ifndef EX2_REPO_SORTBUSLINES_H
#define EX2_REPO_SORTBUSLINES_H
// write only between #define EX2_REPO_SORTBUSLINES_H and #endif //EX2_REPO_SORTBUSLINES_H
#include <string.h>
#define NAME_LEN 21
/**
 * Represents a single bus line with its name and travel details.
 */
typedef struct BusLine
{
    char name[NAME_LEN];
    int distance, duration, frequency;
} BusLine;

typedef enum SortType
{
    DISTANCE,
    DURATION,
    FREQUENCY
} SortType;

/**
 * Sorts the bus lines array by name using the bubble sort algorithm.
 */
void bus_bubble_sort (BusLine *start, BusLine *end);

/**
 * Sorts the bus lines array by a specific type using the quick sort algorithm.
 */
void bus_quick_sort (BusLine *start, BusLine *end, SortType sort_type);

/**
 * Partition function used by quick sort to arrange elements around a pivot.
 */
BusLine *partition (BusLine *start, BusLine *end, SortType sort_type);
// write only between #define EX2_REPO_SORTBUSLINES_H and #endif //EX2_REPO_SORTBUSLINES_H
#endif //EX2_REPO_SORTBUSLINES_H
