#ifndef INTERVAL_H
#define INTERVAL_H 

#include "rtweekend.h"

class interval{
    public: 
        float min;
        float max;

        interval(): min(+infinity), max(-infinity) {}

        interval(float min, float max): min(min), max(max) {}

        float size() const {
            return max - min;
        }

        bool contains(float d){
            return min <= d && d <= max;
        }
        
        bool surrounds(float d){
            return min < d && d < max;
        }

        float clamp(float d){
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