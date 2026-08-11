CC = gcc
CFLAGS = -shared -fPIC -Wall -Wextra
TARGET = libsortbuslines.so
SRCS = financial_engine.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c
	$(CC) -c -fPIC -Wall -Wextra $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
