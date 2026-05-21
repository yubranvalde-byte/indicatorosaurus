#include <stdio.h>
#include <string.h>
#include "sort_bus_lines.h"

void visualize_bus_bubble_sort(BusLine *start, BusLine *end) {
    printf("\033[1;35m### Initial Memory Setup\033[0m\n");
    printf("Array bounds: start = 0x%04lX, end = 0x%04lX\n", 
           0x1000L, 0x1000L + ((char*)end - (char*)start));
    
    int pass = 1;
    for (BusLine *curr = start; curr < end; curr++) {
        printf("\n***\n");
        printf("\033[1;35m### Pass %d: Outer Loop (curr points to 0x%04lX)\033[0m\n", 
               pass++, 0x1000L + ((char*)curr - (char*)start));
        
        int step = 1;
        for (BusLine *curr2 = start; curr2 < start + (end - curr); curr2++) {
            BusLine *next = curr2 + 1;
            
            printf("\n** Step %d.%d: Comparing Index %d and Index %d **\n", 
                   pass - 1, step++, (int)(curr2 - start), (int)(next - start));
            
            // Print current memory layout with pointers
            for (BusLine *ptr = start; ptr <= end; ptr++) {
                const char *name_color = "";
                const char *reset_color = "";
                
                // Color active targets Yellow
                if (ptr == curr2 || ptr == next) {
                    name_color = "\033[33m";
                    reset_color = "\033[0m";
                }
                
                printf("[0x%04lX] %s%-10.20s%s | %2dkm | %2dm | %d/hr", 
                       0x1000L + ((char*)ptr - (char*)start), 
                       name_color, ptr->name, reset_color, 
                       ptr->distance, ptr->duration, ptr->frequency);
                
                // Dynamically build pointer labels string securely
                char labels[256] = "";
                int labels_printed = 0;
                size_t l_size = sizeof(labels);
                if (ptr == start) { strncat(labels, "<-- start", l_size - strlen(labels) - 1); labels_printed++; }
                if (ptr == curr) { strncat(labels, labels_printed ? ", <-- curr (i)" : "<-- curr (i)", l_size - strlen(labels) - 1); labels_printed++; }
                if (ptr == curr2) { strncat(labels, labels_printed ? ", <-- curr2 (j)" : "<-- curr2 (j)", l_size - strlen(labels) - 1); labels_printed++; }
                if (ptr == next) { strncat(labels, labels_printed ? ", <-- next (j+1)" : "<-- next (j+1)", l_size - strlen(labels) - 1); labels_printed++; }
                if (ptr == end) { strncat(labels, labels_printed ? ", <-- end" : "<-- end", l_size - strlen(labels) - 1); labels_printed++; }
                
                if (labels_printed > 0) {
                    printf(" \033[36m%s\033[0m", labels); // Cyan formatting for labels
                }
                
                printf("\n");
            }
            
            printf("* Comparison: strcmp(\"%.20s\", \"%.20s\")\n", next->name, curr2->name);
            
            if (strcmp(next->name, curr2->name) < 0) {
                printf("* Swap Event: \033[32mYES\033[0m. Swapping 0x%04lX and 0x%04lX\n", 
                       0x1000L + ((char*)curr2 - (char*)start), 
                       0x1000L + ((char*)next - (char*)start));
                
                // Perform the actual memory swap
                BusLine placeholder = *next;
                *next = *curr2;
                *curr2 = placeholder;
                
                // Print array state immediately after swap
                printf("* Array State After Swap:\n");
                for (BusLine *ptr = start; ptr <= end; ptr++) {
                    printf("  [0x%04lX] %-10.20s | %2dkm | %2dm | %d/hr\n", 
                           0x1000L + ((char*)ptr - (char*)start), 
                           ptr->name, ptr->distance, ptr->duration, ptr->frequency);
                }
            } else {
                printf("* Swap Event: \033[31mNO\033[0m.\n");
            }
        }
        
        // Educational Lock Note
        BusLine *locked_ptr = start + (end - curr);
        printf("\n>> EDUCATIONAL NOTE: The BusLine \033[32m'%.20s'\033[0m at address 0x%04lX has successfully 'bubbled' to its final position and is now locked from further checks!\n", 
               locked_ptr->name, 0x1000L + ((char*)locked_ptr - (char*)start));
    }
    
    printf("\n***\n\033[1;35m### Final State\033[0m\n");
    for (BusLine *ptr = start; ptr <= end; ptr++) {
        printf("[0x%04lX] %-10.20s | %2dkm | %2dm | %d/hr", 
               0x1000L + ((char*)ptr - (char*)start), 
               ptr->name, ptr->distance, ptr->duration, ptr->frequency);
        
        char labels[256] = "";
        size_t l_size = sizeof(labels);
        if (ptr == start) strncat(labels, "<-- start", l_size - strlen(labels) - 1);
        if (ptr == end)   strncat(labels, labels[0] ? ", <-- end" : "<-- end", l_size - strlen(labels) - 1);
        if (labels[0]) {
            printf(" \033[36m%s\033[0m", labels); // Cyan formatting for labels
        }
        printf("\n");
    }
}
BusLine* visualize_partition(BusLine *start, BusLine *end, SortType sort_type, BusLine *global_start, BusLine *global_end, int depth) {
    BusLine *pivot = end;
    BusLine *i = start;

    printf("\n***\n\033[1;35m### Partition Call (Depth %d): start=0x%04lX, end=0x%04lX\033[0m\n", 
           depth, 0x1000L + ((char*)start - (char*)global_start), 0x1000L + ((char*)end - (char*)global_start));

    int step = 1;
    for (BusLine *curr = start; curr < end; curr++) {
        printf("\n** Step %d: curr points to 0x%04lX **\n", step++, 0x1000L + ((char*)curr - (char*)global_start));

        // Print current memory layout with pointers
        for (BusLine *ptr = global_start; ptr <= global_end; ptr++) {
            const char *name_color = "";
            const char *reset_color = "";
            
            // Color active targets Yellow
            if (ptr == curr || ptr == pivot) {
                name_color = "\033[33m";
                reset_color = "\033[0m";
            }
            
            printf("[0x%04lX] %s%-10.20s%s | %2dkm | %2dm | %d/hr", 
                   0x1000L + ((char*)ptr - (char*)global_start), 
                   name_color, ptr->name, reset_color, 
                   ptr->distance, ptr->duration, ptr->frequency);
            
            char labels[256] = "";
            int labels_printed = 0;
            size_t l_size = sizeof(labels);
            if (ptr == global_start) { strncat(labels, "<-- global_start", l_size - strlen(labels) - 1); labels_printed++; }
            if (ptr == global_end) { strncat(labels, labels_printed ? ", <-- global_end" : "<-- global_end", l_size - strlen(labels) - 1); labels_printed++; }
            if (ptr >= start && ptr <= end) {
                if (ptr == start) { strncat(labels, labels_printed ? ", <-- start" : "<-- start", l_size - strlen(labels) - 1); labels_printed++; }
                if (ptr == i) { strncat(labels, labels_printed ? ", <-- i" : "<-- i", l_size - strlen(labels) - 1); labels_printed++; }
                if (ptr == curr) { strncat(labels, labels_printed ? ", <-- curr" : "<-- curr", l_size - strlen(labels) - 1); labels_printed++; }
                if (ptr == pivot) { strncat(labels, labels_printed ? ", <-- pivot/end" : "<-- pivot/end", l_size - strlen(labels) - 1); labels_printed++; }
                else if (ptr == end && ptr != pivot) { strncat(labels, labels_printed ? ", <-- end" : "<-- end", l_size - strlen(labels) - 1); labels_printed++; }
            }
            if (labels_printed > 0) {
                printf(" \033[36m%s\033[0m", labels);
            }
            printf("\n");
        }

        int swap = 0;
        switch (sort_type) {
            case DISTANCE:
                printf("* Comparison (DISTANCE): %dkm <= %dkm\n", curr->distance, pivot->distance);
                if (curr->distance <= pivot->distance) swap = 1;
                break;
            case DURATION:
                printf("* Comparison (DURATION): %dm <= %dm\n", curr->duration, pivot->duration);
                if (curr->duration <= pivot->duration) swap = 1;
                break;
            case FREQUENCY:
                printf("* Comparison (FREQUENCY): %d/hr <= %d/hr\n", curr->frequency, pivot->frequency);
                if (curr->frequency <= pivot->frequency) swap = 1;
                break;
        }

        if (swap) {
            printf("* Swap Event: \033[32mYES\033[0m. Swapping memory at i (0x%04lX) and curr (0x%04lX)\n", 
                   0x1000L + ((char*)i - (char*)global_start), 
                   0x1000L + ((char*)curr - (char*)global_start));
            
            BusLine placeholder = *i;
            *i = *curr;
            *curr = placeholder;
            i++;
        } else {
            printf("* Swap Event: \033[31mNO\033[0m.\n");
        }
    }

    printf("\n** Final Pivot Swap **\n");
    printf("* Swapping memory at i (0x%04lX) with pivot (0x%04lX)\n", 
           0x1000L + ((char*)i - (char*)global_start), 
           0x1000L + ((char*)pivot - (char*)global_start));

    BusLine placeholder = *i;
    *i = *pivot;
    *pivot = placeholder;

    printf("\n>> EDUCATIONAL NOTE: The BusLine \033[32m'%.20s'\033[0m at address 0x%04lX has successfully been positioned as the partition pivot and is now locked from further checks!\n", 
           i->name, 0x1000L + ((char*)i - (char*)global_start));

    return i;
}

void visualize_bus_quick_sort_recursive(BusLine *start, BusLine *end, SortType sort_type, BusLine *global_start, BusLine *global_end, int depth) {
    if (start < end) {
        BusLine *partition_i = visualize_partition(start, end, sort_type, global_start, global_end, depth);
        visualize_bus_quick_sort_recursive(start, partition_i - 1, sort_type, global_start, global_end, depth + 1);
        visualize_bus_quick_sort_recursive(partition_i + 1, end, sort_type, global_start, global_end, depth + 1);
    }
}

void visualize_bus_quick_sort(BusLine *start, BusLine *end, SortType sort_type) {
    printf("\033[1;35m### Initial Memory Setup (Quick Sort)\033[0m\n");
    printf("Array bounds: start = 0x%04lX, end = 0x%04lX\n", 
           0x1000L, 0x1000L + ((char*)end - (char*)start));
           
    visualize_bus_quick_sort_recursive(start, end, sort_type, start, end, 1);
    
    printf("\n***\n\033[1;35m### Final State (Quick Sort)\033[0m\n");
    for (BusLine *ptr = start; ptr <= end; ptr++) {
        printf("[0x%04lX] \033[32m%-10.20s\033[0m | %2dkm | %2dm | %d/hr", 
               0x1000L + ((char*)ptr - (char*)start), 
               ptr->name, ptr->distance, ptr->duration, ptr->frequency);
        
        char labels[256] = "";
        size_t l_size = sizeof(labels);
        if (ptr == start) strncat(labels, "<-- start", l_size - strlen(labels) - 1);
        if (ptr == end)   strncat(labels, labels[0] ? ", <-- end" : "<-- end", l_size - strlen(labels) - 1);
        if (labels[0]) {
            printf(" \033[36m%s\033[0m", labels);
        }
        printf("\n");
    }
}

