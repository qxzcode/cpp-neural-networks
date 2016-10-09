//
//  NN_test.cpp
//  NN2
//
//  Created by Quinn on 6/9/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//
#include "stdafx.h"

#include "tests.h"

#include <iostream>
#include <vector>
using namespace std;

#include "NeuralNet.h"
using namespace util;

static vector<pair<vec,vec>> randData(size_t num) {
	vector<pair<vec,vec>> data;
	for (size_t n=0; n<num; n++) {
		int b = rand()%8;
		bool b0=(b&1)!=0, b1=(b&2)!=0, b2=(b&4)!=0;
		vec in{(float)b0,(float)b1,(float)b2};
		vec out(8);
		out[b] = 1;
		data.push_back({in,out});
	}
	return data;
}

void NN_test() {
	auto data = randData(100);
	timer t;
	NeuralNet nn(data[0].first.len,data[0].second.len, 1.5);
	constexpr size_t numEpochs = 10000;
	for (size_t epoch=0; epoch<numEpochs; epoch++) {
		//		cout << "------ EPOCH " << (epoch+1) << " ------\n";
		nn.doBatch(data);
	}
	cout << "Took " << t.elapsed() << " seconds\n";
	for (size_t i=0; i<10&&i<data.size(); i++) {
		cout << data[i].first.toString() << "  ";
		cout << nn.process(data[i].first.addEnd()).toString() << "  ";
		cout << data[i].second.toString() << endl;
	}
	cout << nn.m.toString() << endl;
}

