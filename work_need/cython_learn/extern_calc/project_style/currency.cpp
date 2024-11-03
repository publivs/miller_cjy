#include "currency.hh"

namespace ynutil {
    MoneyFormator::MoneyFormator()
    :loc("zh_CN.UTF-8"),
    mnp(std::use_facet<std::money_put<char>>(loc)),
    iterator(os)
    {
        os.imbue(loc);
        os.setf(std::ios_base::showbase);
    }

    MoneyFormator::MoneyFormator(const char* localName)
    :loc(localName),
    mnp(std::use_facet<std::money_put<char>>(loc)),
    iterator(os)
    {
        os.imbue(loc);
        os.setf(std::ios_base::showbase);
    }

    MoneyFormator::~MoneyFormator(){}

    std::string MoneyFormator::str(double value){
        //清理之前遗留的字符流
        os.str("");
        mnp.put(iterator,false,os,' ',value*100.0);
        return os.str();
    }
}