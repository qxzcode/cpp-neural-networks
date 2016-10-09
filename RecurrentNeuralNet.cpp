//
//  RecurrentNeuralNet.cpp
//  NN2
//
//  Created by Quinn on 6/27/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//
#include "stdafx.h"

#include "RecurrentNeuralNet.h"

#include <iostream>
using namespace std;

static std::vector<vec> createEmptyVecs(size_t size, size_t len, bool zero=true) {
	std::vector<vec> v;
	v.reserve(size);
	for (size_t i=0; i<size; i++) {
		if (zero)
			v.emplace_back(len);
		else
			v.emplace_back(len,false);
	}
	return v;
}

static std::vector<mat> createEmptyMats(size_t size, size_t lenX, size_t lenY) {
	std::vector<mat> v;
	v.reserve(size);
	for (size_t i=0; i<size; i++) {
		v.emplace_back(lenX,lenY);
	}
	return v;
}

RecurrentNeuralNet::RecurrentNeuralNet(size_t xL, size_t hL, size_t yL, numf lr):
                                                                        xL(xL),hL(hL),yL(yL),
                                                                        learnRate(lr),
                                                                        cur_h(hL),
                                                                        W_xh(mat::random(xL+1,hL)),dW_xh(mat::random(xL+1,hL)),
	                                                                    W_hh(mat::random(hL+1,hL)),dW_hh(mat::random(hL+1,hL)),
                                                                       	W_hy(mat::random(hL+1,yL)),dW_hy(mat::random(hL+1,yL)) {
	// ...nothing here for now
}

vec RecurrentNeuralNet::step(vec x) {
	// NOTE: not optimized for performance
	cur_h = (W_hh*cur_h.addEnd() + W_xh*x.addEnd()).tanh();
	return (W_hy*cur_h.addEnd()).softmax();
}

void RecurrentNeuralNet::train(std::vector<size_t> x) {
	// Zero the delta accumulators.
	// These will sum up the gradients from
	// each step, to be applied at the end.
	dW_xh.zero();dW_hh.zero();dW_hy.zero();

	size_t numT = x.size()-1;
	vec y(yL,false); // final y, after softmax
	vec h(hL+1,false); h[hL]=1;
	vec hp(hL,false);
	vec prevH(hL+1); prevH[hL]=1; // h[-1] is a vector of all zeros (except for the bias)
	vec yd(yL,false);
	std::vector<mat> prev_hhD = createEmptyMats(hL+1,hL+1,hL);
	std::vector<mat> prev_xhD = createEmptyMats(hL+1,xL+1,hL);
	for (size_t t=0; t<numT; t++) {
		//// FEEDFORWARD ////
		// calculate hp[t] and h[t]
		for (size_t j=0; j<hL; j++) {
			hp[j] = W_xh[x[t]][j]; // select one row (multiplying by a one-hot vector)
			hp[j] += W_xh[xL][j]; // bias unit
			for (size_t i=0; i<hL; i++)
				hp[j] += W_hh[i][j]*prevH[i];
			hp[j] += W_hh[hL][j]; // bias unit
			h[j] = ntanh(hp[j]);
			hp[j] = tanhDeriv(hp[j]);
		}
		// calculate y[t]
		for (size_t j=0; j<yL; j++) {
			numf yp = W_hy[hL][j]; // bias unit
			for (size_t i=0; i<hL; i++) {
				yp += W_hy[i][j]*h[i];
			}
			y[j] = sigmoid(yp);
		}

		// GRADIENT CALCULATIONS //
		// dW_hy
		for (size_t j=0; j<yL; j++) {
			yd[j] = y[j];
			if (j==x[t+1]) yd[j] -= 1;
			yd[j] = 2*yd[j] * sigmoidDeriv(y[j]);

			for (size_t i=0; i<hL; i++) {
				dW_hy[i][j] += yd[j] * h[i];
			}
			dW_hy[hL][j] += yd[j]; // bias unit
		}
		for (size_t j=0; j<hL; j++) {
			// dW_hh
			for (size_t i=0; i<hL+1; i++) {
				numf sumA = 0;
				for (size_t k=0; k<yL; k++) {
					numf sumB = 0;
					for (size_t j2=0; j2<hL; j2++) {
						numf sumC = 0;
						size_t i2m = i==hL?hL+1:hL;
						if (j2==j) sumC += i==hL?1:prevH[i];
						for (size_t i2=0; i2<i2m; i2++) {
							sumC += W_hh[i2][j2]*prev_hhD[i2][i][j];
						}
						prev_hhD[j2][i][j] = hp[j2] * sumC;
						sumB += W_hy[j2][k] * prev_hhD[j2][i][j];
					}
					sumA += yd[k] * sumB;
				}
				dW_hh[i][j] += sumA;
			}
			// dW_xh
			for (size_t i=0; i<xL+1; i++) {
				numf sumA = 0;
				numf xi = numf(i==x[t]||i==xL? 1 : 0);
				for (size_t k=0; k<yL; k++) {
					numf sumB = 0;
					for (size_t j2=0; j2<hL; j2++) {
						numf sumC = 0;
						for (size_t i2=0; i2<hL; i2++) {
							sumC += xi;
							sumC += W_hh[i2][j2]*prev_xhD[i2][i][j];
						}
						prev_xhD[j2][i][j] = hp[j2] * sumC;
						sumB += W_hy[j2][k] * prev_xhD[j2][i][j];
					}
					sumA += yd[k] * sumB;
				}
				dW_xh[i][j] += sumA;
			}
		}

		// prevH = h (just swap for efficiency)
		prevH.swap(h);
	}
	
	W_xh.addWithMult(dW_xh, -learnRate/numT, 2);
	W_hh.addWithMult(dW_hh, -learnRate/numT, 2);
	W_hy.addWithMult(dW_hy, -learnRate/numT, 2);
}

std::string RecurrentNeuralNet::serialize() const {
	std::string str;
	str += "# learning rate\n";
	str += numToStringExact(learnRate)+"\n";
	str += "# xL, hL, yL\n";
	str += std::to_string(xL)+" "+std::to_string(hL)+" "+std::to_string(yL)+"\n";
	str += "\n\n";
	str += "# W_xh\n"+W_xh.serialize()+"\n";
	str += "# W_hh\n"+W_hh.serialize()+"\n";
	str += "# W_hy\n"+W_hy.serialize()+"\n";
	return str;
}
