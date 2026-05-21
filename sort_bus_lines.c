#include "sort_bus_lines.h"

void bus_quick_sort(BusLine *start, BusLine *end, SortType sort_type) {

    if (start < end) {
        BusLine *partition_i = partition(start, end, sort_type);
        bus_quick_sort(start, partition_i-1, sort_type);
        bus_quick_sort(partition_i+1, end, sort_type);
    }
}

BusLine *partition(BusLine *start, BusLine *end, SortType sort_type) {
    BusLine *pivot = end;
    BusLine *i = start;

    for (BusLine *curr = start; curr < end; curr++) {
        int swap = 0;

        switch (sort_type) {
            case DISTANCE:
                if (curr->distance <= pivot->distance) {
                    swap = 1;
                }
                break;
            case DURATION:
                if (curr->duration <= pivot->duration) {
                    swap = 1;
                }
                break;
            case FREQUENCY:
                if (curr->frequency <= pivot->frequency) {
                    swap = 1;
                }
                break;
        }

        // אם הדגל עלה, מבצעים את ההחלפה ומקדמים את pivot_pos
        if (swap) {
            BusLine placeholder = *i;
            *i = *curr;
            *curr = placeholder;

            i++;
        }
    }
    BusLine placeholder = *i;
    *i = *pivot;
    *pivot = placeholder;
    return i;
}

void bus_bubble_sort(BusLine *start, BusLine *end) {

    for (BusLine *curr = start; curr < end; curr++) {
        for (BusLine *curr2 = start; curr2 < start + (end - curr); curr2++) {
            BusLine *next = curr2 + 1;
            if (strcmp(next->name, curr2->name) < 0) {
                BusLine placeholder = *next;
                *next = *curr2;
                *curr2 = placeholder;
            }
        }
    }
}