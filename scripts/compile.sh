#!/bin/bash

# Directory for the compiled library
BUILD_DIR="build"
LIBRARY_NAME="libfortran_code.so"
SRC_DIR="src/fortran"
LIB_PATH="$BUILD_DIR/$LIBRARY_NAME"

# Find all .f95 files in the src/fortran directory
SRC_FILES=$(find "$SRC_DIR" -name "*.f95")

# Check if any .f95 files are found
if [ -z "$SRC_FILES" ]; then
    echo "❌ No .f95 files found in $SRC_DIR."
    exit 1
fi

# Create the 'build' directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating the build directory..."
    mkdir "$BUILD_DIR"
fi

# Start the compilation with gfortran
echo "Compiling the following files:"
echo "$SRC_FILES"

gfortran -shared -fPIC $SRC_FILES -o "$LIB_PATH"

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "✅ Compilation complete. The library is located at $LIB_PATH"
else
    echo "❌ Compilation error. Please check your Fortran files."
    exit 1
fi
