//
//  DeepNeuralNet.cpp
//  NN2
//
//  Created by Quinn on 6/9/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//
#include "stdafx.h"

#include "DeepNeuralNet.h"

DeepNeuralNet::DeepNeuralNet(const std::vector<size_t>& ls, numf lr):lens(ls),ws(lens.size()-1),deltas(ws.size()),learnRate(lr),outs(lens.size()) {
	std::reverse(lens.begin(), lens.end());
	for (size_t n=0; n<ws.size(); n++) {
		ws[n] = mat::random(lens[n+1]+1,lens[n]);
	}
	initStuff();
}

void DeepNeuralNet::initStuff() {
	// find the maximum layer size
	maxLen = 0;
	for (size_t len : lens) {
		if (len>maxLen) maxLen = len;
	}
	
	// pre-allocate deltas
	for (size_t n=0; n<ws.size(); n++) {
		deltas[n] = mat(ws[n].lenX,ws[n].lenY);
	}
	
	// pre-allocate outs
	for (size_t n=0; n<outs.size(); n++) {
		outs[n] = vec(lens[n]+1,false);
	}
	outs[ws.size()][lens[ws.size()]] = 1; // bias unit on the first layer
}

inline void DeepNeuralNet::feedforward(const vec& input) {
	for (size_t x=0; x<input.len; x++) {
		outs[ws.size()][x] = input[x];
	}
	for (long n=ws.size()-1; n>=0; n--) {
		const vec& in = outs[n+1];
		const mat& m = ws[n];
		for (size_t y=0; y<m.lenY; y++) {
			outs[n][y] = 0;
			for (size_t x=0; x<m.lenX; x++) {
				outs[n][y] += in[x]*m[x][y];
			}
			outs[n][y] = sigmoid(outs[n][y]);
		}
		outs[n][m.lenY] = 1;
	}
}

void DeepNeuralNet::doBatch(const std::vector<std::pair<vec,vec>>& data, size_t startI, size_t batchSize) {
	// create & zero deltas
	for (size_t n=0; n<ws.size(); n++) {
		deltas[n].zero();
	}
	
	numf* nextOutD = new numf[maxLen];
	numf* lastOutD = new numf[maxLen];
	size_t nextOutDLen, lastOutDLen;
	
	// loop through each example
	for (size_t i=startI,bn=0; bn<batchSize; bn++,i++) {
		auto& d = data[i%data.size()];
		
		// calculate the outputs of all neurons (feedforward)
		feedforward(d.first);
		
		// go back through and calculate derivatives of all the weights (backpropagation)
		lastOutDLen = d.second.len;
		for (size_t x=0; x<d.second.len; x++) {
			lastOutD[x] = outs[0][x] - d.second[x];
		}
		for (size_t n=0; n<ws.size(); n++) {
			// calculate the derivatives of the next layer's neuron outputs (based on the last)
			// AND
			// calculate the derivative of each weight and add to the delta
			nextOutDLen = lens[n+1];
			const vec& out = outs[n+1];
			for (size_t y=0; y<nextOutDLen; y++) {
				numf sum = 0;
				for (size_t x=0; x<lastOutDLen; x++) {
					numf sdOut = lastOutD[x] * sigmoidDeriv(outs[n][x]);
					if (n<ws.size()-1) // don't need to calculate nextOutD on the last layer
						sum += sdOut * ws[n][y][x];
					
					deltas[n][y][x] -= out[y] * sdOut;
				}
				nextOutD[y] = sum;
			}
			// basically do lastOutD = nextOutD (swap, so no reallocations)
			std::swap(lastOutD, nextOutD);
			lastOutDLen = nextOutDLen;
		}
	}
	
	delete[] nextOutD;
	delete[] lastOutD;
	
	// apply deltas after the entire batch
	numf m = learnRate/batchSize;
	for (size_t n=0; n<ws.size(); n++) {
		ws[n].addWithMult(deltas[n], m);
	}
}

DeepNeuralNet::DeepNeuralNet(std::string file) {
	numReader nr(readFile(file));
	learnRate = nr.readReal();
	lens.resize(nr.readInt());
	ws.resize(lens.size()-1);
	deltas.resize(ws.size());
	outs.resize(lens.size());
	for (size_t n=0; n<lens.size(); n++)
		lens[n] = nr.readInt();
	for (size_t n=0; n<ws.size(); n++) {
		ws[n] = mat(lens[n+1]+1,lens[n]);
		for (size_t x=0; x<ws[n].lenX; x++) {
			for (size_t y=0; y<ws[n].lenY; y++) {
				ws[n][x][y] = nr.readReal();
			}
		}
	}
	initStuff();
}

std::string DeepNeuralNet::serialize() const {
	std::string str;
	str += "# learning rate\n";
	str += numToStringExact(learnRate)+"\n";
	str += "# num layers\n";
	str += std::to_string(lens.size())+"\n";
	str += "# layer sizes\n";
	for (size_t n=0; n<lens.size(); n++)
		str += std::to_string(lens[n])+" ";
	str += "\n\n";
	for (size_t n=0; n<ws.size(); n++) {
		str += "# weights "+std::to_string(n+1)+"-"+std::to_string(n)+"\n";
		for (size_t x=0; x<ws[n].lenX; x++) {
			for (size_t y=0; y<ws[n].lenY; y++) {
				str += numToStringExact(ws[n][x][y])+" ";
			}
			str += "\n";
		}
	}
	return str;
}


