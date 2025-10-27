/**
 * @file Logger.cpp
 * @brief Implementation of Logger static member variables
 * 
 * This file defines the static member variables declared in Logger.h
 * to avoid linker errors (undefined reference).
 */

#include "common/Logger.h"

// Define static member variables
bool Logger::initialized_ = false;
int Logger::currentLogLevel_ = 0;
std::string Logger::logDir_ = "logs";
bool Logger::toConsole_ = true;
bool Logger::toFile_ = true;

