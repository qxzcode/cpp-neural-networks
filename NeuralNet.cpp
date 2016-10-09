//
//  NeuralNet.cpp
//  NN2
//
//  Created by Quinn on 6/8/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//
#include "stdafx.h"

#include "NeuralNet.h"

NeuralNet::NeuralNet(size_t ins, size_t outs, numf lr):m(mat::random(ins+1,outs,-1,1)),learnRate(lr) {
	
}

vec NeuralNet::process(vec& input) {
	return (m*input).sigmoid();
}

void NeuralNet::doBatch(std::vector<std::pair<vec,vec>>& data) {
	mat deltas(m.lenX,m.lenY);
	for (auto& d : data) {
		vec in = d.first.addEnd();
		vec& ans = d.second;
		vec out = process(in);
		
		vec yPart = (out-ans)*2 * out.sigmoidDeriv();
		for (size_t x=0; x<in.len; x++) {
			for (size_t y=0; y<out.len; y++) {
				numf deriv = yPart[y] * in[x];
				deltas[x][y] -= deriv;
			}
			
		}
	}
	m.addWithMult(deltas, learnRate/data.size());
}
