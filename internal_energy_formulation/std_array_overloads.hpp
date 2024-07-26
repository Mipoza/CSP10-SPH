#pragma once
#include <array>
#include <iostream>

/*
 @brief
 Small library for dealing with vector quantities.
 Most things here are pretty self-explanatory
*/

template <typename T, unsigned DIM>
struct Vec: std::array<T, DIM> {
  template <typename... COORDS>
  Vec(COORDS... c): std::array<T, DIM>({static_cast<T>(c)...}){}
  // Assignment
  Vec& operator=(const Vec& other){
    #pragma GCC ivdep
    for(unsigned d = 0; d < DIM; ++d)
      this->operator[](d) = other[d];
    return *this;
  }
  // Assignment
  Vec& operator=(const T& s){
    #pragma GCC ivdep
    for(unsigned d = 0; d < DIM; ++d)
      this->operator[](d) = s;
    return *this;
  }
  // Add-assign
  inline Vec& operator+=(const Vec& other){
    #pragma GCC ivdep
    for(unsigned d = 0; d < DIM; ++d)
      this->operator[](d) += other[d];
    return *this;
  }
  // Sub-assign
  inline Vec& operator-=(const Vec& other){
    #pragma GCC ivdep
    for(unsigned d = 0; d < DIM; ++d)
      this->operator[](d) -= other[d];
    return *this;
  }
  // Unary minus
  inline Vec operator-() const{
    Vec res = *this;
    #pragma GCC ivdep
    for(unsigned d = 0; d < DIM; ++d)
      res[d] *= -1;
    return res;
  }
  // With scalars
  inline Vec operator*=(const T& s){
    #pragma GCC ivdep
    for(unsigned d = 0; d < DIM; ++d)
      this->operator[](d) *= s;
    return *this;
  }
  inline Vec operator/=(const T& s){
    #pragma GCC ivdep
    for(unsigned d = 0; d < DIM; ++d)
      this->operator[](d) /= s;
    return *this;
  }

  // Addition
  inline friend Vec operator+(const Vec& a, const Vec& b){
    Vec res = a;
    res += b;
    return res;
  }
  // Subtraction
  inline friend Vec operator-(const Vec& a, const Vec& b){
    Vec res = a;
    res -= b;
    return res;
  }
  // Multiplication with a scalar
  inline friend Vec operator*(const T& a, const Vec& b){
    Vec res = b;
    #pragma GCC ivdep
    for(unsigned d = 0; d < DIM; ++d)
      res[d] *= a;
    return res;
  }
  inline friend Vec operator*(const Vec& b, const T& a){
    return a*b;
  }
  // Division with a scalar
  inline friend Vec operator/(const Vec& a, const T& b){
    Vec res = a;
    #pragma GCC ivdep
    for(unsigned d = 0; d < DIM; ++d)
      res[d] /= b;
    return res;
  }

  // Dot product
  inline T dot(const Vec& other) const {
    T sum = 0;
    for(unsigned d = 0; d < DIM; ++d)
      sum += this->operator[](d)*other[d];
    return sum;
  }

  // Norm
  inline T norm() const {
    return std::sqrt(this->dot(*this));
  }

  friend std::ostream& operator<<(std::ostream& os, const Vec& v){
    for(unsigned d = 0; d + 1 < DIM; ++d)
      os << v[d] << ", ";
    // The only place where we have -1, make sure no overflows occur
    static_assert(DIM > 0);
    os << v[DIM - 1];
    os << std::endl;
    return os;
  }
};