//
//  DeepNeuralNet.h
//  NN2
//
//  Created by Quinn on 6/9/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//

#pragma once

#include <vector>

#include "util.h"
using namespace util;

class DeepNeuralNet {
public:
	DeepNeuralNet(const std::vector<size_t>& lens, numf lr);
	DeepNeuralNet(std::string file);

	void doBatch(const std::vector<std::pair<vec,vec>>& data) {doBatch(data,0,data.size());}
	void doBatch(const std::vector<std::pair<vec,vec>>& data, size_t startI, size_t batchSize);
	std::string serialize() const;
	
protected:
	friend void DNN_test(); // give access for testing
	
	void initStuff();
	
	inline void feedforward(const vec& input);
	
	// parameters
	std::vector<size_t> lens;
	size_t maxLen;
	std::vector<mat> ws, deltas;
	numf learnRate;
	
	// cache while processing
	std::vector<vec> outs;
	
};
