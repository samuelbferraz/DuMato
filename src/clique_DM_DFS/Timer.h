#ifndef _Timer_H_
#define _Timer_H_

#include <sys/time.h>
#include <iostream>

class Timer
{
	public:
		/*Attributes*/
		timeval start, end;
		double elapsedTime;
		
		/*Functions*/
		Timer();
		void play();
		void pause();
		double getElapsedTimeInSeconds();
		double getElapsedTimeInMiliseconds();
		~Timer();
};
#endif
