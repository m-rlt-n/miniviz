# miniviz build system.
#
# Compiles every csrc/*.c into a single shared library that ctypes loads
# at runtime. The library lives INSIDE the Python package at
# miniviz/_native/, so the same path works in editable installs and
# (eventually) in built wheels.
#
#   - macOS:  miniviz/_native/libminiviz.dylib
#   - Linux:  miniviz/_native/libminiviz.so
#
# Object files are intermediate artifacts; they live in build/ and are
# gitignored. `make clean` removes both.
#
# Usage:
#   make          # build the shared library
#   make test     # build, then run pytest
#   make clean    # remove build artifacts

CC      := cc
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Wpedantic -fPIC
LDFLAGS := -shared

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT := dylib
else
    LIB_EXT := so
endif

SRC_DIR    := csrc
BUILD_DIR  := build
NATIVE_DIR := miniviz/_native
LIB        := $(NATIVE_DIR)/libminiviz.$(LIB_EXT)

SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

.PHONY: all clean test

all: $(LIB)

$(LIB): $(OBJECTS) | $(NATIVE_DIR)
	$(CC) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

$(NATIVE_DIR):
	mkdir -p $@

test: all
	python3 -m pytest tests/ -v

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(NATIVE_DIR)/libminiviz.*
