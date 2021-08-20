# -fPIC process independent code (for shared libraries)
# -Wall to warn about all my bugs :-)
# -std=c99 because I define ints in my loops
# -march=native gives me an extra 10%
CC = cc -Wall -fPIC -shared -std=c99 -march=native -O2
SRC = rams.c
LIB = build/librams.so

all: build ${LIB}
build:
	mkdir -p build

${LIB}: ${SRC} Makefile
	${CC} -o ${LIB} ${SRC}
clean:
	rm -rf build

