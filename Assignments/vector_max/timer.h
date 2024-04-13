#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>
#include <windows.h> 

enum PrintColor { NONE, GREEN, DGREEN, CYAN };

typedef struct {
    LARGE_INTEGER startTime;
    LARGE_INTEGER endTime;
    LARGE_INTEGER frequency;  
} Timer;

static void startTime(Timer* timer) {
    QueryPerformanceFrequency(&timer->frequency);  
    QueryPerformanceCounter(&timer->startTime);    
}

static void stopTime(Timer* timer) {
    QueryPerformanceCounter(&timer->endTime);  
}

static void printElapsedTime(Timer timer, const char* s, enum PrintColor color = NONE) {
    double elapsedTime = ((double)(timer.endTime.QuadPart - timer.startTime.QuadPart)) / timer.frequency.QuadPart;

    // Implement color printing using Windows-specific API functions
    // (Not included here for brevity, but can be added using SetConsoleTextAttribute)

    printf("%s: %.3f ms\n", s, elapsedTime * 1000.0);  
}

#endif
