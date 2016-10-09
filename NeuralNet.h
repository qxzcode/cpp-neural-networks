//
//  NeuralNet.h
//  NN2
//
//  Created by Quinn on 6/8/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//

#pragma once

#include <vector>

#include "util.h"
using namespace util;

class NeuralNet {
public:
	NeuralNet(size_t ins, size_t outs, numf lr);
	
	vec process(vec&& input) {return process(input);}
	vec process(vec& input);
	void doBatch(std::vector<std::pair<vec,vec>>& data);
	
	mat m;
	numf learnRate;
	
};
