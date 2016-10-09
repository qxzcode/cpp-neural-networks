//
//  RNN_test.cpp
//  NN2
//
//  Created by Quinn on 6/27/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//
#include "stdafx.h"

#include "tests.h"

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
using namespace std;

#include "RecurrentNeuralNet.h"
using namespace util;

size_t NUM_SYMS;
static vector<size_t> randData(size_t num) {
	vector<size_t> data;
	srand((unsigned)time(nullptr));
	srand(rand());
	for (size_t n=0; n<num; n++) {
		size_t sym;
		if (n==0) sym = 0;
		else if (n==num-1) sym = 1;
		else sym = 2+n%4%3%2;//rand()%(NUM_SYMS-2)+2;
		data.push_back(sym);
	}
	NUM_SYMS = 4;
	return data;
}
size_t charToSym[128];
char symToChar[128+2];
static vector<size_t> stringData(std::string str) {
	NUM_SYMS = 2;
	for (size_t i=0; i<128; i++) charToSym[i] = 999;
	symToChar[0] = '~';
	symToChar[1] = '\\';
	for (size_t i=0; i<str.size(); i++) {
		char c = str[i];
		if (charToSym[c]==999) {
			size_t sym = NUM_SYMS++;
			charToSym[c] = sym;
			symToChar[sym] = c;
		}
	}
	vector<size_t> data;
	data.reserve(str.size()+2);
	data.push_back(0);
	for (size_t i=0; i<str.size(); i++) {
		char c = str[i];
		data.push_back(charToSym[c]);
	}
	data.push_back(1);
	return data;
}

#include <fstream>
void RNN_test() {
	auto data = stringData("well hello");//"fuzzy wuzzy was a bear. fuzzy wuzzy had no hair. fuzzy wuzzy wasn't fuzzy was he");
	cout << "data:";
	for (auto d : data) {
		cout << " " << d;
	}
	cout << "\n\n";
	timer t;
	RecurrentNeuralNet nn(NUM_SYMS, 10, NUM_SYMS, 0.05f);
	constexpr bool LIMIT_EPOCHS = true;
	constexpr size_t NUM_EPOCHS = 500;

	// learning
	cout << "Starting learning\n";
	double lastPrint = 0.0;
	size_t lastEpochs = 0;
	avgLog<30> perSecLog;
	flushStdin();
	for (size_t epoch=0; (!LIMIT_EPOCHS||epoch<NUM_EPOCHS)&&!isStdinReady(); epoch++) {
		double ela = t.elapsed();
		if (ela-lastPrint > 1.0) { // print a status update every second
			lastPrint += 1.0;
			int percent = int(100.0*epoch/NUM_EPOCHS);
			size_t perSec = epoch-lastEpochs;
			perSecLog.add(perSec);
			size_t secsLeft = (size_t)ceilf((NUM_EPOCHS-epoch)/perSecLog.avg());
			size_t minsLeft = secsLeft/60; secsLeft %= 60;
			size_t hrsLeft = minsLeft/60; minsLeft %= 60;
			cout << "-- Epoch #"<<epoch;
			if (!LIMIT_EPOCHS) cout << ", "<<perSec<<"/sec\n";
			else {
				cout << "/"<<NUM_EPOCHS<<" ("<<percent<<"%), "<<perSec<<"/sec, ~";
				if (hrsLeft>0) cout << hrsLeft<<"hr ";
				if (minsLeft>0 || hrsLeft>0) cout << minsLeft<<"min ";
				if (hrsLeft==0) cout << secsLeft<<"sec ";
				cout << "left\n";
			}
			lastEpochs = epoch;
		}
		nn.train(data);
	}
	flushStdin();
	cout << "\nTook " << t.elapsed() << " seconds\n";
	nn.cur_h.zero();
	vec y = vec::oneHot(0, NUM_SYMS);
	constexpr bool RAND_RES = true;
	if (RAND_RES) cout << "\"";
	for (size_t t=0; t<data.size()-1; t++) {
		y = nn.step(RAND_RES? y : vec::oneHot(data[t], NUM_SYMS));
		struct choice {
			size_t n; numf p;
			bool operator<(const choice& rhs) {
				return p > rhs.p; // sort descending by p
			}
		};
		vector<choice> choices;
		for (size_t i=0; i<y.len; i++) choices.push_back({i,y[i]});
		std::sort(choices.begin(), choices.end());
		if (RAND_RES) {
			double r = (double(rand())/RAND_MAX) * 1.0;
			size_t sym;
			for (auto c : choices) {
				if (r<c.p) {
					sym = c.n;
					break;
				}
				r -= c.p;
			}
			cout << symToChar[sym];
		} else {
			cout << "CHOICE: "<<choices[0].n<<" ("<<int(choices[0].p*100)<<"%)                next: "<<choices[1].n<<" ("<<int(choices[1].p*100)<<"%)\n";
			cout << "ACTUAL: "<<data[t+1]<<endl;
			cout << "    "<<y.toString()<<endl;
		}
	}
	if (RAND_RES) cout << "\"\n";
	cout << nn.dW_hy.toString() << endl;

	// save network to file
	cout << "\nSaving network to file\n";
	ofstream fs;
	std::string file = "net_"+std::to_string(time(nullptr))+".txt";
	fs.open(dataDir+file);
	fs << nn.serialize();
	fs.close();

	// save debug log
	ofstream dbgFile;
	dbgFile.open("C:/Users/Quinn.Tucker18/Desktop/dbg.txt", ios::out | ios::trunc);
	dbgFile << dbg.str();
	dbgFile.close();
}

