# Compiler to use
CC = g++

# Compiler flags
CFLAGS = -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Wconversion \
         -Wnull-dereference -Wsign-conversion -std=c++17

# Linker flags
LDFLAGS = -lgmp -lgmpxx

# Target executable name
TARGET = main.out

# Source files
SOURCES = main.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# MISC
RED = \033[0;31m
RESET = \033[0m

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo -e "executable $(RED)$(TARGET)$(RESET) generated"

# Compile source files to object files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(TARGET)

# Phony targets
.PHONY: all clean
