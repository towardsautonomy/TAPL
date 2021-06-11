/**
 * @file    taplLog.hpp
 * @brief   This file provides logging functions for TAPL
 * @author  Shubham Shrivastava
 */

#include "date.h"

#ifndef TAPL_LOG_H_
#define TAPL_LOG_H_

/**< Color Codes */
#define RESET       "\033[0m"
#define BLACK       "\033[30m"             // Black 
#define RED         "\033[31m"             // Red 
#define GREEN       "\033[32m"             // Green 
#define YELLOW      "\033[33m"             // Yellow 
#define BLUE        "\033[34m"             // Blue 
#define MAGENTA     "\033[35m"             // Magenta 
#define CYAN        "\033[36m"             // Cyan 
#define WHITE       "\033[37m"             // White 
#define BOLDBLACK   "\033[1m\033[30m"      // Bold Black 
#define BOLDRED     "\033[1m\033[31m"      // Bold Red 
#define BOLDGREEN   "\033[1m\033[32m"      // Bold Green 
#define BOLDYELLOW  "\033[1m\033[33m"      // Bold Yellow 
#define BOLDBLUE    "\033[1m\033[34m"      // Bold Blue 
#define BOLDMAGENTA "\033[1m\033[35m"      // Bold Magenta
#define BOLDCYAN    "\033[1m\033[36m"      // Bold Cyan 
#define BOLDWHITE   "\033[1m\033[37m"      // Bold White 

/**
 * @brief Log level enumerations 
 */
enum LogLevelEnum {
    DEBUG_LOG_LEVEL,                /**< Debug Log */
    INFO_LOG_LEVEL,                 /**< Info Log */
    WARN_LOG_LEVEL,                 /**< Warning Log */
    ERROR_LOG_LEVEL,                /**< Error Log */
    NOTICE_LOG_LEVEL,               /**< Notice Log */
    FATAL_LOG_LEVEL                 /**< Fatal Log */
};

class Log
{
public:
    Log(const std::string &fileName,
        const std::string &funcName,
        const int &lineNum, 
        const enum LogLevelEnum &logLevel) {

        // get color code based on debug level
        std::string colorCode;
        switch(logLevel) {
        case DEBUG_LOG_LEVEL:
            colorCode = CYAN;
            std::cout << colorCode << "[DEBUG]";
            break;
        case INFO_LOG_LEVEL:
            colorCode = GREEN;
            std::cout << colorCode << "[INFO]";
            break;
        case WARN_LOG_LEVEL:
            colorCode = BOLDYELLOW;
            std::cout << colorCode << "[WARN]";
            break;
        case ERROR_LOG_LEVEL:
            colorCode = BOLDRED;
            std::cout << colorCode << "[ERROR]";
            break;
        case NOTICE_LOG_LEVEL:
            colorCode = BOLDYELLOW;
            std::cout << colorCode << "[NOTICE]";
            break;
        case FATAL_LOG_LEVEL:
            colorCode = RED;
            std::cout << colorCode << "[FATAL]";
            break;
        default:
            colorCode = "";
        }

        // get time string
        std::string datetimeStr = date::format("%F %T", std::chrono::system_clock::now());
        std::cout << "[" << datetimeStr << "]"           \
                     "[" << "file:" << fileName << "|" \
                         << "func:" << funcName << "|" \
                         << "line:" << lineNum << "] || ";
    }

    template <class T>
    Log &operator<<(const T &v) {
        std::cout << v;
        return *this;
    }

    ~Log() {
        std::cout << RESET << std::endl;
    }
};

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

/**< Debug Log */
#define TLOG_DEBUG Log(__FILENAME__, __FUNCTION__, __LINE__, DEBUG_LOG_LEVEL)
/**< Info Log */
#define TLOG_INFO Log(__FILENAME__, __FUNCTION__, __LINE__, INFO_LOG_LEVEL)
/**< Warning Log */
#define TLOG_WARN Log(__FILENAME__, __FUNCTION__, __LINE__, WARN_LOG_LEVEL)
/**< Error Log */
#define TLOG_ERROR Log(__FILENAME__, __FUNCTION__, __LINE__, ERROR_LOG_LEVEL)
/**< Notice Log */
#define TLOG_NOTICE Log(__FILENAME__, __FUNCTION__, __LINE__, NOTICE_LOG_LEVEL)
/**< Fatal Log */
#define TLOG_FATAL Log(__FILENAME__, __FUNCTION__, __LINE__, FATAL_LOG_LEVEL)

#endif /* TAPL_LOG_H_ */