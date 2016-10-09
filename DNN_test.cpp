//
//  DNN_test.cpp
//  NN2
//
//  Created by Quinn on 6/9/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//
#include "stdafx.h"

#include "tests.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <algorithm>
using namespace std;

#include "DeepNeuralNet.h"
using namespace util;

numf rreal() {
	return numf(rand())/RAND_MAX;
}

static vector<pair<vec,vec>> randData(size_t num) {
	vector<pair<vec,vec>> data;
	for (size_t n=0; n<num; n++) {
		int a=rand()%2, b=rand()%2, c=rand()%2;
		vec in{(float)a,(float)b,(float)c};
		vec out{float(a != b != c)};
		data.push_back({in,out});
	}
	return data;
}

static vector<pair<vec,vec>> irisData(bool train=true) {
	vector<pair<vec,vec>> data;
	numReader nr(readFile(dataDir+"iris.data"));
	for (size_t n=0; n<150; n++) {
		vec in{
			(nr.readReal()-4.3f)/3.6f,
			(nr.readReal()-2.0f)/2.4f,
			(nr.readReal()-1.0f)/5.9f,
			(nr.readReal()-0.1f)/2.4f
		};
		int o = nr.readInt();
		vec out{float(o==0),float(o==1),float(o==2)};
		if (n%2==(int)train)
			data.push_back({in,out});
	}
	return data;
}

static vector<pair<vec,vec>> wineData(bool train=true) {
	vector<pair<vec,vec>> data;
	numReader nr(readFile(dataDir+"wine.data"));
	vec mins(13);
	vec maxes(13);
	for (size_t i=0; i<13; i++)
		mins[i] = numeric_limits<numf>::max();
	for (size_t n=0; n<178; n++) {
		int o = nr.readInt();
		vec in(13);
		for (size_t i=0; i<in.len; i++) {
			in[i] = nr.readReal();
			if (in[i]<mins[i]) mins[i] = in[i];
			if (in[i]>maxes[i]) maxes[i] = in[i];
		}
		vec out{float(o==1),float(o==2),float(o==3)};
		if (n%2==(int)train)
			data.push_back({in,out});
	}
	for (size_t n=0; n<data.size(); n++) {
		for (size_t i=0; i<13; i++) {
			data[n].first[i] = (data[n].first[i]-mins[i])/(maxes[i]-mins[i]);
		}
	}
	return data;
}

static vector<pair<vec,vec>> cancerData(bool train=true) {
	vector<pair<vec,vec>> data;
	numReader nr(readFile(dataDir+"cancer.data"));
	vec mins(30);
	vec maxes(30);
	for (size_t i=0; i<30; i++)
		mins[i] = numeric_limits<numf>::max();
	for (size_t n=0; n<569; n++) {
		int id=nr.readInt(), o=nr.readInt();
		vec in(30);
		for (size_t i=0; i<in.len; i++) {
			in[i] = nr.readReal();
			if (in[i]<mins[i]) mins[i] = in[i];
			if (in[i]>maxes[i]) maxes[i] = in[i];
		}
		vec out{float(o)};
		if (n%2==(int)train)
			data.push_back({in,out});
	}
	for (size_t n=0; n<data.size(); n++) {
		for (size_t i=0; i<30; i++) {
			data[n].first[i] = (data[n].first[i]-mins[i])/(maxes[i]-mins[i]);
		}
	}
	return data;
}

static vector<pair<vec,vec>> heartData(bool train=true) {
	vector<pair<vec,vec>> data;
	numReader nr(readFile(dataDir+"heart.data"));
	vec mins(13);
	vec maxes(13);
	for (size_t i=0; i<13; i++)
		mins[i] = numeric_limits<numf>::max();
	for (size_t n=0; n<303; n++) {
		vec in(13);
		for (size_t i=0; i<in.len; i++) {
			in[i] = nr.readReal();
			if (in[i]<mins[i]) mins[i] = in[i];
			if (in[i]>maxes[i]) maxes[i] = in[i];
		}
		numf o = nr.readReal()/4.0f;
		vec out{o};
		if (n%2==(int)train)
			data.push_back({in,out});
	}
	for (size_t n=0; n<data.size(); n++) {
		for (size_t i=0; i<13; i++) {
			data[n].first[i] = (data[n].first[i]-mins[i])/(maxes[i]-mins[i]);
		}
	}
	return data;
}

//#define TESTING
#define dataFunc cancerData

void DNN_test() {
	// setup
	cout << "Loading data...\n";
	auto data = dataFunc();
	random_shuffle(data.begin(), data.end());
#ifndef TESTING
	timer t;
	DeepNeuralNet nn({data[0].first.len,10,10,data[0].second.len}, 20.0);
	constexpr size_t NUM_EPOCHS = 400000;
	constexpr size_t BATCH_SIZE = 50;
	
	// learning
	cout << "Starting learning\n";
	double lastPrint = 0.0;
	size_t lastEpochs = 0;
	avgLog<20> perSecLog;
	size_t dataI = 0;
	flushStdin();
	for (size_t epoch=0; epoch<NUM_EPOCHS&&!isStdinReady(); epoch++) {
		double ela = t.elapsed();
		if (ela-lastPrint > 1.0) { // print a status update every second
			lastPrint += 1.0;
			int percent = int(100.0*epoch/NUM_EPOCHS);
			size_t perSec = epoch-lastEpochs;
			perSecLog.add(perSec);
			size_t secsLeft = (size_t)ceilf((NUM_EPOCHS-epoch)/perSecLog.avg());
			size_t minsLeft = secsLeft/60; secsLeft %= 60;
			size_t hrsLeft = minsLeft/60; minsLeft %= 60;
			cout << "-- Epoch #"<<epoch<<"/"<<NUM_EPOCHS<<" ("<<percent<<"%), "<<perSec<<"/sec, ~";
			if (hrsLeft>0) cout << hrsLeft<<"hr ";
			if (minsLeft>0 || hrsLeft>0) cout << minsLeft<<"min ";
			if (hrsLeft==0) cout << secsLeft<<"sec ";
			cout << "left\n";
			lastEpochs = epoch;
		}
		nn.doBatch(data,dataI,BATCH_SIZE);
		dataI += BATCH_SIZE;
	}
	flushStdin();
#else
	DeepNeuralNet nn(dataDir+"heart net.txt");
#endif
	
	// console results
#ifndef TESTING
	double elapsed = t.elapsed();
	cout << "\nTook " << elapsed << " seconds\n";
#endif
	numf avgErr=0, minErr=numeric_limits<numf>::max(), maxErr=0;
	data = dataFunc(false);
	for (size_t i=0; i<data.size(); i++) {
		vec& in = data[i].first;
		vec& ans = data[i].second;
		nn.feedforward(in);
		vec out = nn.outs[0].remEnd();
		
		numf err = (out-ans).abs().sum() / out.len; // avg. deviation (abs difference)
		avgErr += err;
		if (err<minErr) minErr = err;
		if (err>maxErr) maxErr = err;
		
		if (i%(data.size()/10)==0) {
			cout << in.toString() << "  ";
			//cout << nn.outs[1].toString() << "  ";
			cout << out.toString() << "  ";
			cout << ans.toString() << "  ";
			cout << util::numToString(err*100) << "%\n";
		}
	}
	avgErr /= data.size();
	cout << "Max. error: " << util::numToString(maxErr*100) << "%\n";
	cout << "Avg. error: " << util::numToString(avgErr*100) << "%\n";
	cout << "Min. error: " << util::numToString(minErr*100) << "%\n";
	/*for (size_t n=nn.ws.size(); n>0; n--) {
		cout << "MAT "<<n<<"-"<<(n-1) << endl;
		cout << nn.ws[n-1].toString() << endl;
	}//*/
	
#ifndef TESTING
	// save network to file
	cout << "\nSaving network to file\n";
	ofstream fs;
	std::string file = "net_"+std::to_string(time(nullptr))+".txt";
	fs.open(dataDir+file);
	fs << nn.serialize();
	fs.close();
#endif
}

