
CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

SOURCES = main.c data_utils.c preprocessing.c metrics.c logistic_regression.c linear_regression.c knn.c decision_tree.c naive_bayes.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = ml_program

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
