#include <stdio.h>
#include <stdlib.h>
#include "sort_bus_lines.h"
#include "test_bus_lines.h"
#define BUFFER_SIZE 61
#define EXPECTED_ARGS 2
#define EXPECTED_FIELDS 4
#define MIN_DISTANCE 0
#define MAX_DISTANCE 1000
#define MIN_DURATION 10
#define MAX_DURATION 100
#define MIN_FREQ 1
#define MAX_FREQ 50
#define NAME_TEST 2
#define DURATION_TEST 4
#define DISTANCE_TEST 6
#define FREQ_TEST 8

int is_valid_line(BusLine *curr) {
    char *ptr = curr->name;
    while (*ptr != '\0') {
        if (!(('0' <= *ptr && *ptr <= '9') || ('a' <= *ptr && *ptr <= 'z'))) {
            printf("Error: invalid input. name should only have lower-case letters and digits. try again:\n");
            return 0;
        }
        ptr++;
    }
    if (curr->distance < MIN_DISTANCE || curr->distance > MAX_DISTANCE) {
        printf("Error: invalid input. distance should be between 0-1000. try again:\n");
        return 0;
    }
    if (curr->duration < MIN_DURATION || curr->duration > MAX_DURATION) {
        printf("Error: invalid input. duration should be between 10-100. try again:\n");
        return 0;
    }
    if (curr->frequency < MIN_FREQ || curr->frequency > MAX_FREQ) {
        printf("Error: invalid input. frequency should be between 1-50. try again:\n");
        return 0;
    }
    return 1;
}

BusLine* get_user_input(int *num_lines) {
    // קליטת כמות הקווים
    // הקצאת המערך lines
    // לולאת ה-for לקליטת ואימות פרטי כל קו
    // return lines;
    char buffer[BUFFER_SIZE];

    while (1) {
        printf("Enter number of lines. Then enter\n");
        if (fgets(buffer,sizeof(buffer), stdin) == NULL) {
            printf("Error: trouble reading the input. try again.\n");
            continue;
        }
        if (sscanf(buffer, "%d", num_lines) != 1) {
            printf("Error: invalid input. try again:\n");
            continue;
        }
        if (*num_lines <= 0) {
            printf("Error: NO input. try again:\n");
            continue;
        }
        break;
    }
    BusLine *lines = malloc(sizeof(BusLine) * (*num_lines));
    if (lines == NULL) {
        return NULL;
    }

    for (int i = 0; i < *num_lines; i++) {

        while (1) {
            printf("Enter line info. Then enter\n");
            if (fgets(buffer,sizeof(buffer), stdin) == NULL) {
                printf("Error: trouble reading the input. try again:\n");
                continue;
            }
            BusLine *curr = lines + i;
            if (sscanf(buffer, "%20[^,],%d,%d,%d", curr->name, &curr->distance, &curr->duration, &curr->frequency) != EXPECTED_FIELDS) {
                printf("Error: invalid input. try again:\n");
                continue;
            }
            if (!is_valid_line(curr)) {
                continue;
            }
            break;

        }
    }
    return lines;
}

void check_and_print_equality(BusLine *lines, BusLine *original, int num_lines, int test_num, int *failed) {
    if (is_equal(lines, lines + num_lines - 1, original, original + num_lines - 1)) {
        printf("TEST %d PASSED: equal to original\n", test_num);
    } else {
        printf("TEST %d FAILED: not equal to original\n", test_num);
        (*failed)++;
    }
}

void restore_array(BusLine *lines, BusLine *original, int num_lines) {
    for (int i = 0; i < num_lines; i++) {
        *(lines + i) = *(original + i);
    }
}

int run_test_mode(BusLine *lines, int num_lines) {
    BusLine* original_copy = malloc(sizeof(BusLine)* num_lines);
    if (original_copy == NULL) {
        return EXIT_FAILURE;
    }
    for (int i = 0; i < num_lines; i++) {
        *(original_copy + i) = *(lines + i);
    }
    int failed = 0;
    bus_bubble_sort(lines, lines + num_lines - 1);
    if (is_sorted_by_name(lines, lines + num_lines - 1)) {printf("TEST 1 PASSED: sorted by name\n");} else {printf("TEST 1 FAILED: not sorted by name\n"); failed++;}
    check_and_print_equality(lines, original_copy, num_lines, NAME_TEST, &failed);
    restore_array(lines, original_copy, num_lines);
    bus_quick_sort(lines, lines + num_lines - 1, DURATION);
    if (is_sorted_by_duration(lines, lines + num_lines - 1)) {printf("TEST 3 PASSED: sorted by duration\n");} else {printf("TEST 3 FAILED: not sorted by duration\n"); failed++;}
    check_and_print_equality(lines, original_copy, num_lines, DURATION_TEST, &failed);
    restore_array(lines, original_copy, num_lines);
    bus_quick_sort(lines, lines + num_lines - 1, DISTANCE);
    if (is_sorted_by_distance(lines, lines + num_lines - 1)) {printf("TEST 5 PASSED: sorted by distance\n");} else {printf("TEST 5 FAILED: not sorted by distance\n"); failed++;}
    check_and_print_equality(lines, original_copy, num_lines, DISTANCE_TEST, &failed);
    restore_array(lines, original_copy, num_lines);
    bus_quick_sort(lines, lines + num_lines - 1, FREQUENCY);
    if (is_sorted_by_frequency(lines, lines + num_lines - 1)) {printf("TEST 7 PASSED: sorted by frequency\n");} else {printf("TEST 7 FAILED: not sorted by frequency\n"); failed++;}
    check_and_print_equality(lines, original_copy, num_lines, FREQ_TEST, &failed);
    restore_array(lines, original_copy, num_lines);
    free(original_copy);
    return (failed > 0) ? EXIT_FAILURE: EXIT_SUCCESS;
}

/**
 * Main program for bus line management.
 * Reads lines from input and sorts them or runs tests based on arguments.
 */
int main (int argc, char *argv[]) {
    if (argc != EXPECTED_ARGS) {
        printf("Usage: arguments amount is wrong. please insert the following: '$./sort_lines <sort category>'\n");
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1],"by_name") != 0 && strcmp(argv[1], "by_duration") != 0 && strcmp(argv[1], "by_distance") != 0 && strcmp(argv[1],"by_frequency") != 0 && strcmp(argv[1], "test") != 0) {
        printf("Usage: command is wrong. please insert the following: '$./sort_lines <sort_category>', when sort_category must be one of: by_duration, by_distance, by_frequency, by_name, test\n");
        return EXIT_FAILURE;
    }
    int num_lines = 0;
    BusLine *lines = get_user_input(&num_lines);

    if (lines == NULL) {
        return EXIT_FAILURE; // טיפול במקרה שהקצאת הזיכרון נכשלה בתוך הפונקציה
    }

    if (strcmp(argv[1], "test") == 0) {
        if (run_test_mode(lines, num_lines) == EXIT_FAILURE) {
            free(lines);
            return EXIT_FAILURE;
        }
    } else {
        if (strcmp(argv[1], "by_name") == 0) {
            bus_bubble_sort(lines, lines + num_lines - 1);
        } else if (strcmp(argv[1], "by_distance") == 0) {
            bus_quick_sort(lines, lines + num_lines - 1, DISTANCE);
        } else if (strcmp(argv[1], "by_duration") == 0) {
            bus_quick_sort(lines, lines + num_lines - 1, DURATION);
        } else if (strcmp(argv[1], "by_frequency") == 0) {
            bus_quick_sort(lines, lines + num_lines - 1, FREQUENCY);
        }
        for (BusLine *curr = lines; curr < lines + num_lines; curr++) {
            printf("%s,%d,%d,%d\n", curr->name, curr->distance, curr->duration, curr->frequency);
        }
    }
    free(lines);
    return EXIT_SUCCESS;
}
