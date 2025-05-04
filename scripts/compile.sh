#!/bin/bash

# Config
BUILD_DIR="build"
LIBRARY_NAME="libfortran_code.so"
LIB_PATH="$BUILD_DIR/$LIBRARY_NAME"

CLASSIC_DIR="src/fortran/classic"
PARALLEL_DIR="src/fortran/parallelized"

# Find all .f95 files
CLASSIC_FILES=$(find "$CLASSIC_DIR" -name "*.f95")
PARALLEL_FILES=$(find "$PARALLEL_DIR" -name "*.f95")

ALL_FILES="$CLASSIC_FILES $PARALLEL_FILES"

# Check if files exist
if [ -z "$ALL_FILES" ]; then
    echo "❌ No .f95 files found in classic or parallelized folders."
    exit 1
fi

# Create build directory if needed
if [ ! -d "$BUILD_DIR" ]; then
    echo "📁 Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Compile everything together, enabling OpenMP (won’t hurt classic code)
echo "🔧 Compiling all sources into one library..."
echo "$ALL_FILES"

gfortran -fopenmp -shared -fPIC $ALL_FILES -o "$LIB_PATH"

# Check result
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful. Library created at: $LIB_PATH"
else
    echo "❌ Compilation failed."
    exit 1
fi
