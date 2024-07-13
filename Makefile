.PHONY = all clean

CC = gcc
MKDIR = mkdir -p
FLAGS = -ggdb -Wall -Wextra -lm -fopenmp -rdynamic -I./include/
HDRS = include/*.h src/tests/test_utils.h
ODIR = build
SDIR = src
SRCS = $(shell find $(SDIR) -name "*.c" -not -path "src/tests/*" -not -path "src/examples/*") src/tests/test_utils.c
OBJS = ${SRCS:$(SDIR)/%.c=$(ODIR)/%.o}

$(shell $(MKDIR) build)
$(shell $(MKDIR) build/tests)
$(shell $(MKDIR) build/examples)

all: tests examples

# examples
examples: xor

xor: src/examples/xor.c $(OBJS) $(SRCS) $(HDRS)
	$(CC) -o $@ src/examples/xor.c $(OBJS) $(FLAGS)
# tests
tests: mat_tests nn_tests

nn_tests: src/tests/nn_tests.c $(OBJS) $(SRCS) $(HDRS)
	$(CC) -o $@ src/tests/nn_tests.c $(OBJS) $(FLAGS)

mat_tests: src/tests/mat_tests.c $(OBJS) $(SRCS) $(HDRS)
	$(CC) -o $@ src/tests/mat_tests.c $(OBJS) $(FLAGS)

$(ODIR)/%.o: $(SDIR)/%.c $(HDRS)
	$(CC) -o $@ $< -c $(FLAGS)

clean:
	rm -f ${OBJS}
