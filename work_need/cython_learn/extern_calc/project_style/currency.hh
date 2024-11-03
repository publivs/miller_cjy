#ifndef MONEYFORMATOR_H
#define MONEYFORMATOR_H
#include <iostream>
#include <iterator>
#include <locale>
#include <string>
#include <sstream>

namespace ynutil{
    class MoneyFormator{
    public:
        MoneyFormator();
        MoneyFormator(const char*);
        ~MoneyFormator();

        std::string str(double);

    private:
        std::locale loc;
        const std::money_put<char>& mnp;
        std::ostringstream os;
        std::ostreambuf_iterator<char,std::char_traits<char>> iterator;
    };
}
#endif