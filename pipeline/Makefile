CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c99 -I../phase5 -I../phase6
LDFLAGS = -lm

TARGET = pipeline.exe

OBJS = main.o transformer.o quant.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.c ../phase5/transformer.h ../phase6/quant.h
	$(CC) $(CFLAGS) -c -o $@ main.c

transformer.o: ../phase5/transformer.c ../phase5/transformer.h
	$(CC) $(CFLAGS) -c -o $@ ../phase5/transformer.c

quant.o: ../phase6/quant.c ../phase6/quant.h
	$(CC) $(CFLAGS) -c -o $@ ../phase6/quant.c

clean:
	rm -f $(OBJS) $(TARGET) model.bin

.PHONY: all clean
