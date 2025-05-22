#ifndef INTERVAL_H
#define INTERVAL_H 

#include "rtweekend.h"

class interval{
    public: 
        double min;
        double max;

        interval(): min(+infinity), max(-infinity) {}

        interval(double min, double max): min(min), max(max) {}

        double size() const {
            return max - min;
        }

        bool contains(double d){
            return min <= d && d <= max;
        }
        
        bool surrounds(double d){
            return min < d && d < max;
        }

        double clamp(double d){
            if (d < min) return min;
            if (d > max) return max;
            return d;
        }

        static const interval empty;
        static const interval universe;
};

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

#endif