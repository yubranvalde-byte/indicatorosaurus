CC = gcc
CFLAGS = -shared -fPIC -O2 -Wall -Wextra
TARGET = libsortbuslines.so
SRCS = financial_engine.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

%.o: %.c
	$(CC) -c -fPIC -O2 -Wall -Wextra $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
