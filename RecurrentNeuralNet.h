//
//  RecurrentNeuralNet.h
//  NN2
//
//  Created by Quinn on 6/27/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//
#pragma once

#include <vector>

#include "util.h"
using namespace util;

class RecurrentNeuralNet {
public:
	RecurrentNeuralNet(size_t xL, size_t hL, size_t yL, numf lr);

	vec step(vec x);
	void train(std::vector<size_t> seq);
	std::string serialize() const;

protected:
	friend void RNN_test();

	size_t xL, hL, yL;
	numf learnRate;

	vec cur_h;
	mat W_xh, W_hh, W_hy;
	mat dW_xh, dW_hh, dW_hy; // pre-allocate this memory for performance reasons

};