//
//  util.h
//  NN2
//
//  Created by Quinn on 6/7/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//

#pragma once

#include <string>
#include <sstream>
#include <cmath>
#include <ctime>

#ifdef _DEBUG
#include <exception>
#define ASSERT_I(idx,max) if((idx)>=(max))throw std::runtime_error("subscript out of range");
#else
#define ASSERT_I(idx,max) ;
#endif // _DEBUG

extern std::ostringstream dbg;
using std::endl;

namespace util {
	
	typedef float numf;
	
	inline float nsqrt(float x) {return sqrtf(x);}
	inline double nsqrt(double x) {return sqrt(x);}
	inline long double nsqrt(long double x) {return sqrtl(x);}
	inline float nabs(float x) {return fabsf(x);}
	inline double nabs(double x) {return fabs(x);}
	inline long double nabs(long double x) {return fabsl(x);}
	inline float nlog(float x) {return logf(x);}
	inline double nlog(double x) {return log(x);}
	inline long double nlog(long double x) {return logl(x);}
	inline float nexp(float x) {return expf(x);}
	inline double nexp(double x) {return exp(x);}
	inline long double nexp(long double x) {return expl(x);}
	inline float ntanh(float x) {return tanhf(x);}
	inline double ntanh(double x) {return tanh(x);}
	inline long double ntanh(long double x) {return tanhl(x);}
	inline float ncosh(float x) {return coshf(x);}
	inline double ncosh(double x) {return cosh(x);}
	inline long double ncosh(long double x) {return coshl(x);}
	inline numf tanhDeriv(numf x) { // d/dx tanh(x) = sech(x)^2
		x = ncosh(x);
		return 1 / (x*x);
	}
	inline numf sigmoid(numf x) {
		return 1 / (1+nexp(-x));
	}
	inline numf sigmoidDeriv(numf y) {
		return y*(1-y);
	}
	
	std::string numToString(numf x);
	std::string numToStringExact(numf x);
	
	std::string readFile(std::string file);
	
	struct numReader {
		numReader(std::string str);
		
		numf readReal();
		size_t readInt();
		
		std::istringstream ss;
	};
	
	struct vec {
		/// basic/default constructor
		vec(size_t l = 0):v(l==0? nullptr : new numf[l]),len(l) {
			for (size_t i=0; i<len; i++)
				v[i] = 0;
		}
		vec(size_t l, bool zero):v(new numf[l]),len(l) {
//			if (zero)
//				for (size_t i=0; i<len; i++)
//					v[i] = 0;
		}
		/// initializer list constructor
		vec(std::initializer_list<numf> il):v(il.size()==0? nullptr : new numf[il.size()]),len(il.size()) {
			for (size_t i=0; i<len; i++)
				v[i] = il.begin()[i];
		}
		/// copy constructor
		vec(const vec& other):v(new numf[other.len]),len(other.len) {
			for (size_t i=0; i<len; i++)
				v[i] = other[i];
		}
		/// move constructor
		vec(vec&& other):v(other.v),len(other.len) {
			other.v = nullptr;
			other.len = 0;
		}
		/// copy assignment operator
		vec& operator=(const vec& rhs) {
			if (this != &rhs) {
				delete[] v;
				len = rhs.len;
				v = new numf[len];
				for (size_t i=0; i<len; i++)
					v[i] = rhs[i];
			}
			return *this;
		}
		/// move assignment operator
		vec& operator=(vec&& rhs) {
			if (this != &rhs) {
				delete[] v;
				len = rhs.len;
				v = rhs.v;
				rhs.len = 0;
				rhs.v = nullptr;
			}
			return *this;
		}
		/// destructor
		~vec() {
			delete[] v;
		}
		
		/// subscript operator
		numf& operator[](size_t i) {ASSERT_I(i,len);return v[i];}
		const numf& operator[](size_t i) const {ASSERT_I(i,len);return v[i];}
		
		/// add/remove element at end (for a bias)
		vec addEnd(numf val = 1) const {
			vec res(len+1);
			for (size_t i=0; i<len; i++)
				res[i] = v[i];
			res[len] = val;
			return res;
		}
		vec remEnd() const {
			vec res(len-1);
			for (size_t i=0; i<len-1; i++)
				res[i] = v[i];
			return res;
		}
		
		/// math
		vec operator+(const vec& rhs) const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = v[i]+rhs[i];
			return res;
		}
		vec operator-(const vec& rhs) const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = v[i]-rhs[i];
			return res;
		}
		vec operator*(const vec& rhs) const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = v[i]*rhs[i];
			return res;
		}
		vec operator/(const vec& rhs) const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = v[i]/rhs[i];
			return res;
		}
		vec operator+(float rhs) const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = v[i]+rhs;
			return res;
		}
		vec operator-(float rhs) const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = v[i]-rhs;
			return res;
		}
		vec operator*(float rhs) const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = v[i]*rhs;
			return res;
		}
		vec operator/(float rhs) const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = v[i]/rhs;
			return res;
		}
		vec square() const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = v[i]*v[i];
			return res;
		}
		vec abs() const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = nabs(v[i]);
			return res;
		}
		vec sigmoid() const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = util::sigmoid(v[i]);
			return res;
		}
		vec sigmoidDeriv() const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = util::sigmoidDeriv(v[i]);
			return res;
		}
		vec tanh() const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = util::ntanh(v[i]);
			return res;
		}
		vec softmax() const {
			vec res(len);
			numf sum = 0;
			for (size_t i=0; i<len; i++)
				sum += (res[i] = nexp(v[i]));
			for (size_t i=0; i<len; i++)
				res[i] /= sum;
			return res;
		}
		numf crossEntropyLoss(vec ans) const {
			numf sum = 0;
			for (size_t i=0; i<len; i++) {
				if (ans[i]!=0)
					sum += ans[i] * util::nlog(v[i]);
			}
			return -sum;
		}
		template<class T>
		vec apply(T func) const {
			vec res(len);
			for (size_t i=0; i<len; i++)
				res[i] = func(v[i]);
			return res;
		}
		numf sum() const {
			numf res = 0;
			for (size_t i=0; i<len; i++)
				res += v[i];
			return res;
		}
		numf magSq() const {
			numf res = 0;
			for (size_t i=0; i<len; i++)
				res += v[i]*v[i];
			return res;
		}
		numf mag() const {
			return nsqrt(magSq());
		}
		vec& zero() {
			for (size_t i=0; i<len; i++)
				v[i] = 0;
			return *this;
		}
		void swap(vec& o) {
			std::swap(v,o.v);
			std::swap(len,o.len);
		}

		static vec oneHot(size_t i, size_t len) {
			vec v(len);
			v[i] = 1;
			return v;
		}
		
		/// display functions
		std::string toString() const {
			std::string s = "[";
			for (size_t i=0; i<len; i++) {
				if (i!=0) s += ",";
				s += numToString(v[i]);
			}
			s += "]";
			return s;
		}
		
		/// data members
		numf* v;
		size_t len;
	};
	
	struct mat {
		/// basic/default constructor
		mat(size_t x = 0, size_t y = 0):m((x==0||y==0)? nullptr : new numf[x*y]),lenX(y==0?0:x),lenY(x==0?0:y) {
			for (size_t i=0; i<lenX*lenY; i++)
				m[i] = 0;
		}
		/// copy constructor
		mat(const mat& other):m(new numf[other.lenX*other.lenY]),lenX(other.lenX),lenY(other.lenY) {
			for (size_t i=0; i<lenX*lenY; i++)
				m[i] = other.m[i];
		}
		/// move constructor
		mat(mat&& other):m(other.m),lenX(other.lenX),lenY(other.lenY) {
			other.m = nullptr;
			other.lenX = 0;
			other.lenY = 0;
		}
		/// copy assignment operator
		mat& operator=(const mat& rhs) {
			if (this != &rhs) {
				delete[] m;
				lenX = rhs.lenX;
				lenY = rhs.lenY;
				m = new numf[lenX*lenY];
				for (size_t i=0; i<lenX*lenY; i++)
					m[i] = rhs.m[i];
			}
			return *this;
		}
		/// move assignment operator
		mat& operator=(mat&& rhs) {
			if (this != &rhs) {
				delete[] m;
				lenX = rhs.lenX;
				lenY = rhs.lenY;
				m = rhs.m;
				rhs.m = nullptr;
				rhs.lenX = 0;
				rhs.lenY = 0;
			}
			return *this;
		}
		/// destructor
		~mat() {
			delete[] m;
		}
		
		/// subscript operator
#ifdef _DEBUG
	// bounds-checking wrappers around a pointer-to-numf
	struct matRow_t {
		numf& operator[](size_t y) {ASSERT_I(y,lenY);return p[y];}
		numf* p;
		size_t lenY;
	};
	struct matRowC_t {
		const numf& operator[](size_t y) {ASSERT_I(y,lenY);return p[y];}
		const numf* p;
		size_t lenY;
	};
#define makeMatRow(ptr) {(ptr),lenY}
#else
	typedef numf* matRow_t;
	typedef const numf* matRowC_t;
#define makeMatRow(ptr) (ptr)
#endif // _DEBUG
		matRow_t operator[](size_t x) {ASSERT_I(x,lenX);return makeMatRow(m+(x*lenY));}
		matRowC_t operator[](size_t x) const {ASSERT_I(x,lenX);return makeMatRow(m+(x*lenY));}
		
		/// math
		vec operator*(const vec& rhs) const {
			vec res(lenY);
			for (size_t x=0; x<lenX; x++) {
				for (size_t y=0; y<lenY; y++) {
					res[y] += rhs[x]*m[x*lenY+y];
				}
			}
			return res;
		}
		mat operator+(const mat& rhs) const {
			mat res(lenX,lenY);
			for (size_t x=0; x<lenX; x++) {
				for (size_t y=0; y<lenY; y++) {
					size_t i = x*lenY+y;
					res.m[i] = m[i]+rhs.m[i];
				}
			}
			return res;
		}
		mat operator-(const mat& rhs) const {
			mat res(lenX,lenY);
			for (size_t x=0; x<lenX; x++) {
				for (size_t y=0; y<lenY; y++) {
					size_t i = x*lenY+y;
					res.m[i] = m[i]-rhs.m[i];
				}
			}
			return res;
		}
		mat& addWithMult(const mat& other, numf f) {
			for (size_t x=0; x<lenX; x++) {
				for (size_t y=0; y<lenY; y++) {
					size_t i = x*lenY+y;
					m[i] += other.m[i]*f;
				}
			}
			return *this;
		}
		mat& addWithMult(const mat& other, numf f, numf cap) {
			for (size_t x=0; x<lenX; x++) {
				for (size_t y=0; y<lenY; y++) {
					size_t i = x*lenY+y;
					numf d = other.m[i]*f;
					if (d>cap) d=cap;
					if (d<-cap) d=-cap;
					m[i] += d;
				}
			}
			return *this;
		}
		void zero() {
			for (size_t x=0; x<lenX; x++) {
				for (size_t y=0; y<lenY; y++) {
					m[x*lenY+y] = 0;
				}
			}
		}
		
		/// display functions
		std::string toString() const {
			std::string s = "[";
			for (size_t x=0; x<lenX; x++) {
				s += "[";
				for (size_t y=0; y<lenY; y++) {
					if (y!=0) s += ",";
					s += numToString(m[x*lenY+y]);
				}
				if (x!=lenX-1) s += "],\n ";
				else s += "]";
			}
			s += "]";
			return s;
		}
		std::string serialize() const {
			std::string str;
			for (size_t x=0; x<lenX; x++) {
				for (size_t y=0; y<lenY; y++) {
					str += numToStringExact((*this)[x][y])+" ";
				}
				str += "\n";
			}
			return str;
		}
		
		/// data
		numf* m;
		size_t lenX, lenY;
		
		/// static random factory function
		static mat random(size_t xl, size_t yl) {
			numf m = 1/nsqrt((numf)xl);
			return random(xl, yl, -m, m);
		}
		static mat random(size_t xl, size_t yl, numf min, numf max);
	};
	
	struct timer {
		timer():begin(std::clock()) {}
		
		void reset() {begin=std::clock();}
		double elapsed() const {return double(std::clock()-begin)/CLOCKS_PER_SEC;}
		
		clock_t begin;
	};

	template<size_t LEN>
	struct avgLog {
		template<class T>
		void add(T x) {
			if (first) {
				for (size_t i=0; i<LEN; i++)
					log[i] = (float)x;
				first = false;
			} else {
				log[nextI++] = (float)x;
				nextI %= LEN;
			}
		}
		float avg() {
			float sum = 0;
			for (size_t i=0; i<LEN; i++)
				sum += log[i];
			return sum/LEN;
		}
		
		float log[LEN];
		size_t nextI = 0;
		bool first = true;
	};

	void flushStdin();
	bool isStdinReady();

	const std::string dataDir = "C:/Users/Quinn.Tucker18/Documents/Neural Network data/";
	
}

