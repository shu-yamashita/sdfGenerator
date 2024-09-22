#ifndef INCLUDED_SDFGENERATOR_COMMON_LOGGER_H
#define INCLUDED_SDFGENERATOR_COMMON_LOGGER_H

#include <cstdio>
#include <string>

namespace sdfGenerator
{
namespace common
{

class processLogger
{
private:
	const std::string str_;
public:
	processLogger( const std::string& str ): str_(str)
	{
		printf("[sdfGenerator]: %s started.\n", str_.c_str());
	}
	~processLogger()
	{
		printf("[sdfGenerator]: %s ended.\n", str_.c_str());
	}
};

} // namespace common
} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_COMMON_LOGGER_H
