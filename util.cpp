//
//  util.cpp
//  NN2
//
//  Created by Quinn on 6/7/16.
//  Copyright (c) 2016 Hexahedron Games. All rights reserved.
//
#include "stdafx.h"

#include "util.h"

#include <random>
#include <fstream>
#include <iomanip>
#include <float.h>
#include <limits>

std::ostringstream dbg;

static std::stringstream initSS(bool exact) {
	std::stringstream ss;
	if (exact) {
		ss << std::setprecision(
#ifdef FLT_DECIMAL_DIG
			FLT_DECIMAL_DIG
#else
			FLT_DIG+3
#endif
		);
	} else {
		ss << std::fixed << std::setprecision(2);
	}
	return ss;
}
std::stringstream ss = initSS(false);
std::string util::numToString(numf x) {
	ss.str(std::string());
	ss << x;
	return ss.str();
}

std::stringstream sse = initSS(true);
std::string util::numToStringExact(numf x) {
	sse.str(std::string());
	sse << x;
	return sse.str();
}

std::string util::readFile(std::string file) {
	std::ifstream fs(file);
	std::string str, file_contents;
	while (std::getline(fs, str)) {
		file_contents += str;
		file_contents.push_back('\n');
	}
	return file_contents;
}

util::numReader::numReader(std::string str):ss(str) {
	
}

template<class T>
static T readNum(std::istringstream& ss) {
	T num;
	while(ss >> num || !ss.eof()) {
		if(ss.fail()) {
			ss.clear();
			std::string str;
			ss >> str;
			if (str[0]=='#') {
				// skip to the next line
				ss.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
			}
			continue;
		}
		return num;
	}
	throw std::runtime_error("EOF in readNum");
}

util::numf util::numReader::readReal() {
	return readNum<numf>(ss);
}

size_t util::numReader::readInt() {
	return readNum<size_t>(ss);
}

std::random_device rd;
std::mt19937 gen(rd());
util::mat util::mat::random(size_t xl, size_t yl, numf min, numf max) {
	std::uniform_real_distribution<numf> dist(min,max);
	mat m(xl,yl);
	for (size_t x=0; x<xl; x++) {
		for (size_t y=0; y<yl; y++) {
			m.m[x*yl+y] = dist(gen);
		}
	}
	return m;
}

#include <Windows.h>
void util::flushStdin() {
	FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
}
bool util::isStdinReady() {
	DWORD numEvents;
	GetNumberOfConsoleInputEvents(GetStdHandle(STD_INPUT_HANDLE), &numEvents);
	if (numEvents==0) return false;
	INPUT_RECORD record;
	ReadConsoleInput(GetStdHandle(STD_INPUT_HANDLE), &record, 1, &numEvents);
	KEY_EVENT_RECORD& ke = record.Event.KeyEvent;
	return record.EventType==KEY_EVENT && ke.bKeyDown && ke.wVirtualKeyCode==VK_RETURN;
}
