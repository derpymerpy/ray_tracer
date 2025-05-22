#ifndef VEC3_H
#define VEC3_H

#include "rtweekend.h"

class vec3 {
    public: 
        double e[3];

        vec3() : e{0, 0, 0} {}
        vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

        double x() const{
            return e[0];
        }
        double y() const{
            return e[1];
        }
        double z() const{
            return e[2];
        }

        vec3 operator-() const {return vec3(-e[0], -e[1], -e[2]); };
        double operator[](int i) const {return e[i]; };
        double& operator[](int i) {return e[i]; };

        vec3& operator+=(const vec3& v){
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        vec3& operator*=(double c){
            e[0] *= c;
            e[1] *= c;
            e[2] *= c;
            return *this;
        }

        vec3& operator/=(double c){
            return (*this)*=(1/c);
        }

        double length_squared() const{
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }

        double length() const{
            return std::sqrt(length_squared());
        }

        bool near_zero() const{
            auto s = 1e-18;
            return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
        }
};

using point3 = vec3;

inline std::ostream& operator<<(std::ostream& out, const vec3& v){
    return out << v.e[0] <<" " << v.e[1] << " "<<v.e[2];
}

inline vec3 operator+(const vec3& u, const vec3& v){
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3& u, const vec3& v){
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3& u, const vec3& v){
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

//scalar multiplication
inline vec3 operator*(const vec3& u, double t){
    return vec3(u.e[0] * t, u.e[1] * t, u.e[2] *t);
}

inline vec3 operator*(double t, const vec3& u){
    return u * t;
}

inline vec3 operator/(const vec3& u, double t){
    return u * (1/t);
}

inline double dot(const vec3& u, const vec3& v){
    return (u.e[0] * v.e[0] 
          + u.e[1] * v.e[1] 
          + u.e[2] * v.e[2]);
}

inline vec3 cross(const vec3& u, const vec3& v){
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1], 
                u.e[2] * v.e[0] - u.e[0] * v.e[2], 
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3 unit_vector(const vec3& v){
    return v/ (v.length());
}

static vec3 random_vec(){
    return vec3(random_double(), random_double(), random_double());
}

static vec3 random_vec(double min, double max){
    return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
}

inline vec3 random_in_unit_disk(){
    while (true){
        vec3 v = vec3(random_double(-1, 1), random_double(-1, 1), 0.0);
        if(v.length_squared() < 1){
            return v;
        }
    }
}


inline vec3 random_unit_vec(){
    while (true) {
        auto p = random_vec(-1.0, 1.0);
        double l = p.length_squared();
        //reject vectors outside sphere to ensure uniform distribution
        if (1e-60 < l && l <= 1){
            return p/sqrt(l);
        }
    }
}

inline vec3 random_on_hemisphere(const vec3& norm){
    vec3 v = random_unit_vec();
    if (dot(v, norm) > 0) return v;
    return -v;
}

inline vec3 reflect(const vec3& v_in, const vec3& norm){
    //negative since the dot product already flips the sign 
    return v_in - 2*norm*dot(v_in, norm);
}

inline vec3 refract(const vec3& v_in, const vec3& norm, double n_frac){
    //expects v_in and norm to be unit vectors
    double cos = std::fmin(dot(-v_in, norm), 1.0);
    vec3 r_perp = n_frac * (v_in + cos * norm);
    vec3 r_parallel = - sqrt(fabs(1.0 - r_perp.length_squared())) * norm;
    return r_perp + r_parallel;
}

#endif