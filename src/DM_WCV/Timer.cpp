#include "Timer.h"

Timer::Timer()
{
	elapsedTime = 0;
}

void Timer::play()
{
	gettimeofday(&start, NULL);
}

void Timer::pause()
{
	gettimeofday(&end, NULL);
	elapsedTime += (end.tv_sec - start.tv_sec) * 1000.0;     // sec to ms
	elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;   // us to ms
}

void Timer::play(const char *message)
{
	printf("[Timer][BEGIN]: %s\n", message);
	gettimeofday(&start, NULL);
}

void Timer::pause(const char *message)
{
	gettimeofday(&end, NULL);
	elapsedTime += (end.tv_sec - start.tv_sec) * 1000.0;     // sec to ms
	elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;   // us to ms
	printf("[Timer][END]: %s\n", message);
}

double Timer::getElapsedTimeInMiliseconds()
{
	return elapsedTime;
}

double Timer::getElapsedTimeInSeconds()
{
	return elapsedTime / 1000;
}

Timer::~Timer()
{

}
